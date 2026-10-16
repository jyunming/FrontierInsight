"""Repair of the code reader's findings: test first, one finding at a time, never a version that breaks something.

The reader finds real faults, and a whole-file rewrite for a batch of them fixed few and broke others. Here each issue
(:mod:`core.code_findings`: a finding by a stable id) is taken on its own, the ones that can change a reported number or
figure first:

1. **A failing test.** The code model is asked for a small test that fails, by an assertion, on the CURRENT code because
   of the finding (:data:`TEST_HEAD`). FI runs it twice. If it passes, the finding is *not demonstrated*: it is recorded
   as such and never repaired (a repair would chase the reader's wording). A test that fails for any other reason (it
   raised, did not import) shows nothing and is asked for again, within its bound.
2. **A repair of the function(s) involved**, not of the file (:data:`REPAIR_HEAD`): the reply's functions are spliced into
   the file (:func:`splice`), same names and same parameters; a finding that names no function is repaired in its file
   as a whole, and the file must keep everything it had (:func:`core.code_review.repair_problem`).
3. **Accepted only when nothing else broke.** The new test passes, every test that passed before still passes (the
   equation tests, every earlier accepted finding test), no static check has more faults than before, and the file
   keeps what FI reads from it. Otherwise the previous text is put back: a version that broke something is never kept.
4. **Bounded.** Each issue has a few tries of each kind (:data:`TEST_CALLS`, :data:`REPAIR_CALLS`) and the quest a
   total (:data:`core.code_review.MAX_FINDING_CALLS`); every model request is counted on disk BEFORE it is made, so a
   resume never asks again. A function that still fails after its tries may be rewritten ONCE by the other model
   configured for review (:data:`OTHER_CALLS`), and the record says which model fixed it.

Nothing here knows the engine: the model, the test runs and the regression snapshot are passed in (:class:`Env`).
"""

from __future__ import annotations

import ast
import copy
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from . import code_findings as _issues
from . import function_steps as _steps
from . import optimise as _optimise
from . import trial_runner as _trials

#: Where a test written for an issue is kept (under the quest root).
TEST_DIR = Path(".fi") / "finding_tests"
#: Per issue, in all its rounds: requests for a test, requests for a repair by the code model, requests to the other model.
TEST_CALLS = 3
REPAIR_CALLS = 3
OTHER_CALLS = 1
#: The most functions one reply may replace in a file.
MAX_FUNCTIONS = 3
#: Issues that change no reported result are repaired only while the quest has used fewer requests than this.
MINOR_LIMIT = 30
#: The openings of the two requests (``agents/code_finding_test.md``, ``agents/code_review_repair.md``).
TEST_HEAD = "# A Test That Shows One Finding"
REPAIR_HEAD = "# Code That Does Not Do What the Plan Requires"
MARK = "FINDING_TESTS:"
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

#: Run in the quest's environment: imports each test file with ``code/`` on the path and calls its ``test_*`` functions.
#: Only an ``AssertionError`` raised by the test's own check of values is a failure (``kind: assertion``, with the line of
#: the check). Anything else is an error that shows nothing, with the place it was raised (``place``: ``program`` when the
#: deepest frame in the quest's files is under ``code/``, ``test`` when it is the test file) and its ``file:line``: an
#: exception (``kind: exception``), an exception the test caught and turned into an assertion (``wrapped``; it is judged by
#: the exception it caught), and an assertion the program's own code raised (``program_assert``).
HARNESS = '''
import importlib.util, json, os, sys, traceback
code_dir = os.path.abspath({code!r})
sys.path.insert(0, code_dir)

def where(e, path):
    place, at, line = "library", "", ""
    for f in traceback.extract_tb(e.__traceback__):
        fn = os.path.abspath(f.filename)
        if fn == path:
            place, at, line = "test", os.path.basename(fn) + ":" + str(f.lineno), f.line or ""
        elif fn.startswith(code_dir + os.sep):
            place, at, line = "program", "code/" + os.path.relpath(fn, code_dir).replace(os.sep, "/") + ":" + str(f.lineno), f.line or ""
    return place, at, line[:200]

def said(e):
    return (type(e).__name__ + ": " + str(e))[:300]

def judged(name, e, path):
    if isinstance(e, AssertionError):
        caught = e.__cause__ or e.__context__
        if caught is not None and not isinstance(caught, AssertionError):
            place, at, line = where(caught, path)
            return {{"test": name, "status": "error", "kind": "wrapped", "place": place, "at": at, "line": line,
                     "detail": said(caught) + " (the test turned it into an assertion)"}}
        place, at, line = where(e, path)
        if place != "program":
            return {{"test": name, "status": "fail", "kind": "assertion", "place": place, "at": at, "line": line,
                     "detail": str(e)[:300]}}
        return {{"test": name, "status": "error", "kind": "program_assert", "place": place, "at": at, "line": line,
                 "detail": said(e)}}
    place, at, line = where(e, path)
    return {{"test": name, "status": "error", "kind": "exception", "place": place, "at": at, "line": line,
             "detail": said(e)}}

out = {{}}
for given in {paths!r}:
    path = os.path.abspath(given)
    results = []
    try:
        spec = importlib.util.spec_from_file_location("finding_test_" + str(len(out)), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except BaseException as e:
        results.append(judged("(import)", e, path))
    else:
        names = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
        if not names:
            results.append({{"test": "(none)", "status": "error", "kind": "exception", "place": "test",
                             "detail": "the file defines no test_* function"}})
        for n in names:
            try:
                getattr(mod, n)()
            except BaseException as e:
                results.append(judged(n, e, path))
            else:
                results.append({{"test": n, "status": "pass", "detail": ""}})
    out[given] = results
print({mark!r} + json.dumps(out))
'''


#: What a script reads from the environment FI gives a run: the variable, the file it names (the paths the run itself uses,
#: relative to the quest folder) and what a stand-in holds until the run has written the real one.
_RUN_FILES = (("FI_TRIALS", Path(_trials.RAW_DIRNAME) / _trials.SUMMARY_NAME, '{"cells": []}'),
              ("FI_OPTIMISATION", _optimise.LEDGER_PATH, ""),
              ("FI_BEST_DESIGN", _optimise.BEST_PATH, "{}"))
STAND_IN_DIR = TEST_DIR / "_env"


def run_environment(root: Path, raw_dir: str = _trials.RAW_DIRNAME, *, stand_ins: bool = True) -> dict[str, str]:
    """The ``FI_*`` variables a finding test runs with, the ones the real run gives the code: the real file when the run
    has written it. Without it, a stand-in under ``.fi/finding_tests/_env/`` (never under ``raw/`` or ``results/``, never over
    a real file) when ``stand_ins`` is true, so a module that reads one while it is imported does not fail on a variable that
    is not set; a test run after the real run (``stand_ins=False``) gets only real files, and a missing one makes it error."""
    env: dict[str, str] = {}
    if not all(runs_nothing_on_import(f) for f in sorted((root / "code").glob("*.py"))):
        # importing such a script would run the analysis again in the quest folder (and could rewrite its results):
        # it gets no FI_* variables, as before, so the import stops at the first one it reads
        return env
    for var, rel, content in _RUN_FILES:
        if (root / rel).is_file():
            env[var] = rel.as_posix()
            continue
        stand = STAND_IN_DIR / rel.name
        try:
            if stand_ins:
                (root / stand).parent.mkdir(parents=True, exist_ok=True)
                if not (root / stand).is_file():
                    (root / stand).write_text(content, encoding="utf-8")
                env[var] = stand.as_posix()
        except OSError:
            pass  # the test then runs as it did before: an import that needs the variable shows as an error
    if (root / raw_dir).is_dir():
        env["FI_RAW_DIR"] = raw_dir
    elif stand_ins:
        try:
            (root / STAND_IN_DIR / "raw").mkdir(parents=True, exist_ok=True)
            env["FI_RAW_DIR"] = (STAND_IN_DIR / "raw").as_posix()
        except OSError:
            pass
    return env


def runs_nothing_on_import(path: Path) -> bool:
    """Whether importing the script runs nothing: at its top level only imports, definitions, assignments that call
    nothing, a docstring and an ``if __name__ == "__main__":`` block. A file that cannot be read or parsed does not."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return False
    from .code_layout import _is_main_guard

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if _is_main_guard(stmt) and not stmt.orelse:
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and not any(isinstance(n, ast.Call) for n in ast.walk(stmt)):
            continue
        return False
    return True


def harness_source(code_dir: str, paths: list[str]) -> str:
    return HARNESS.format(code=str(code_dir), paths=[str(p) for p in paths], mark=MARK)


@dataclass
class TestRun:
    """What running one test file gave: ``pass`` (every test passed), ``fail`` (a test's own check of values failed by an
    assertion, nothing else went wrong) or ``error`` (it could not be run: it did not import, raised something else, turned
    an exception into an assertion, or printed nothing). ``kind`` is the harness's (``assertion``, ``exception``,
    ``wrapped``, ``program_assert``), ``place`` where it was raised (``program``: the quest's code; ``test``: the test file),
    ``at`` its ``file:line`` and ``line`` that line's text (for a failure: the assertion that failed)."""

    status: str
    detail: str = ""
    kind: str = ""
    place: str = ""
    at: str = ""
    line: str = ""

    @property
    def crash_in_program(self) -> bool:
        """An error raised inside the quest's own code (not the test's): a crash, for the crash repair."""
        return self.status == "error" and self.place == "program"


def parse_runs(stdout: str, paths: list[str]) -> dict[str, TestRun]:
    """``{path: TestRun}`` from the harness's output; every path asked for is there, an error when nothing was read."""
    data: dict[str, Any] = {}
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(MARK):
            try:
                loaded = json.loads(line[len(MARK):])
            except ValueError:
                break
            data = loaded if isinstance(loaded, dict) else {}
            break
    out: dict[str, TestRun] = {}
    for path in paths:
        rows = data.get(str(path))
        if not isinstance(rows, list) or not rows:
            out[path] = TestRun("error", "the test could not be run here")
            continue
        errors = [r for r in rows if isinstance(r, dict) and r.get("status") == "error"]
        fails = [r for r in rows if isinstance(r, dict) and r.get("status") == "fail"]
        first = errors[0] if errors else fails[0] if fails else None
        if first is None:
            out[path] = TestRun("pass")
            continue
        at = str(first.get("at") or "")
        detail = (f"{first.get('test')}: {first.get('detail') or 'the assertion failed'}"
                  + (f" (at {at})" if at and errors else ""))
        out[path] = TestRun("error" if errors else "fail", detail, str(first.get("kind") or ""),
                            str(first.get("place") or ""), at, str(first.get("line") or ""))
    return out


def regressions(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """The checks that passed before and do not now."""
    return sorted(k for k, v in before.items() if v == "pass" and after.get(k) != "pass")


def worse_static(before: dict[str, int], after: dict[str, int]) -> list[str]:
    """The static checks that have more faults than before."""
    return sorted(k for k, n in after.items() if n > before.get(k, 0))


# --- what the code model returns -----------------------------------------------------------------------------------------


def reply_test(reply: str) -> str:
    """The test file in a reply: the first fenced block that parses and defines a ``test_*`` function, else ``""``."""
    blocks = [m.group(1) for m in _FENCE.finditer(reply or "")] or [reply or ""]
    for text in blocks:
        text = text.strip("\n") + "\n"
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        if any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") for n in tree.body):
            return text
    return ""


def _top_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def params_kept(old: ast.FunctionDef | ast.AsyncFunctionDef, new: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Why a repaired function's parameters are not allowed (``""`` when they are): every parameter it had stays, by the
    same name, kind and place, so every call that worked still works; a parameter may be added only after them and only
    with a default value (a positional one at the end, or a keyword-only one), and ``*args`` / ``**kwargs`` may be added."""
    a, b = old.args, new.args
    want = _steps._param_names(old)

    def says() -> str:
        return (f"must keep its parameters ({', '.join(want)}), adding only parameters with a default value after them; "
                f"the reply has ({', '.join(_steps._param_names(new))})")

    old_pos, new_pos = [x.arg for x in [*a.posonlyargs, *a.args]], [x.arg for x in [*b.posonlyargs, *b.args]]
    if new_pos[:len(old_pos)] != old_pos or len(b.posonlyargs) != len(a.posonlyargs):
        return says()
    added = len(new_pos) - len(old_pos)
    # defaults belong to the last positional parameters: each added one needs one, and the old ones keep theirs
    if added and len(b.defaults) < added + len(a.defaults):
        return says()
    if (a.vararg and (not b.vararg or b.vararg.arg != a.vararg.arg)) or (a.kwarg and (not b.kwarg or b.kwarg.arg != a.kwarg.arg)):
        return says()
    old_kw = {x.arg for x in a.kwonlyargs}
    for x, default in zip(b.kwonlyargs, b.kw_defaults):
        if x.arg not in old_kw and default is None:
            return says()
    if not old_kw <= {x.arg for x in b.kwonlyargs}:
        return says()
    return ""


def direct_callers(code: Path, rel: str, name: str) -> dict[str, set[str]]:
    """``{file under code/: the top-level functions in it that call `name` directly}`` (``name(...)`` or ``module.name(...)``),
    in every Python file of ``code/`` but its tests; ``name`` itself is not its own caller."""
    out: dict[str, set[str]] = {}
    for path in sorted(code.rglob("*.py")):
        other = path.relative_to(code).as_posix()
        if any(part in ("tests", "__pycache__") for part in path.relative_to(code).parts[:-1]):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        for fname, fn in _top_functions(tree).items():
            if other == rel and fname == name:
                continue
            for node in ast.walk(fn):
                func = node.func if isinstance(node, ast.Call) else None
                if (isinstance(func, ast.Name) and func.id == name) or (isinstance(func, ast.Attribute) and func.attr == name):
                    out.setdefault(other, set()).add(fname)
                    break
    return out


def unpassed_params(old_fn: str, new_fn: str, name: str, callers: dict[str, str], caller_names: dict[str, set[str]]) -> str:
    """Why a repair that added parameters to ``name`` leaves the study on their defaults (``""`` when it does not): every
    call of ``name`` in a function that calls it directly (``callers``: ``{file: its text after the repair}``;
    ``caller_names``: the calling functions of each file) must pass each added parameter, by name or by place (or forward
    ``**kwargs``). A parameter added with the old value as its default and passed only by the finding's own test changes
    nothing the study runs."""
    try:
        old = _top_functions(ast.parse(old_fn)).get(name)
        new = _top_functions(ast.parse(new_fn)).get(name)
    except SyntaxError:
        return ""
    if old is None or new is None:
        return ""
    before = set(_steps._param_names(old))
    positional = [x.arg for x in [*new.args.posonlyargs, *new.args.args]]
    added = [p for p in _steps._param_names(new) if p not in before and p not in (
        new.args.vararg.arg if new.args.vararg else None, new.args.kwarg.arg if new.args.kwarg else None)]
    if not added:
        return ""
    for other, text in sorted(callers.items()):
        try:
            funcs = _top_functions(ast.parse(text))
        except SyntaxError:
            continue
        for fname in sorted(caller_names.get(other) or ()):
            fn = funcs.get(fname)
            for node in ast.walk(fn) if fn is not None else ():
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if not ((isinstance(f, ast.Name) and f.id == name) or (isinstance(f, ast.Attribute) and f.attr == name)):
                    continue
                if any(k.arg is None for k in node.keywords):
                    continue  # **kwargs passed on
                given = {k.arg for k in node.keywords}
                places = len(node.args) if not any(isinstance(a, ast.Starred) for a in node.args) else len(positional)
                missing = [p for p in added if p not in given and not (p in positional and positional.index(p) < places)]
                if missing:
                    return (f"the study's call still uses the default: `{fname}` in {other} calls `{name}` without "
                            + ", ".join(f"`{p}`" for p in missing)
                            + ", so the repair changes nothing the study runs; pass it on from the caller")
    return ""


def _calls_to(node: ast.AST, name: str) -> list[ast.Call]:
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and (
        (isinstance(c.func, ast.Name) and c.func.id == name) or (isinstance(c.func, ast.Attribute) and c.func.attr == name))]


def _typed_number(node: ast.AST) -> bool:
    """Whether an argument is only a number written into the code (``0.01``, ``-1e-3``, ``(2)``)."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _typed_number(node.operand)
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) and not isinstance(
        node.value, bool)


def caller_edit_problem(before: str, after: str, caller_names: set[str] | Iterable[str], callee: str) -> str:
    """Why a repair's change to a function that calls the repaired one directly is not allowed (``""`` when it is). Only the
    arguments of its calls of ``callee`` may change: everything else in the caller stays as it was (compared as code, with
    those calls' arguments set aside), so a repair cannot change another reported number on its way. And an argument the
    repair adds or changes may not be a number typed into the call: the old typed-in value passed by name is the same
    typed-in value, and the study would run unchanged."""
    try:
        old_funcs, new_funcs = _top_functions(ast.parse(before)), _top_functions(ast.parse(after))
    except SyntaxError:
        return ""

    def masked(fn: ast.FunctionDef) -> str:
        fn = copy.deepcopy(fn)
        for call in _calls_to(fn, callee):
            call.args, call.keywords = [], []
        return ast.dump(fn)

    for name in sorted(caller_names):
        old, new = old_funcs.get(name), new_funcs.get(name)
        if old is None or new is None or ast.dump(old) == ast.dump(new):
            continue
        if masked(old) != masked(new):
            return (f"it changed `{name}` beyond its call of `{callee}`: only the arguments of its call of `{callee}` may "
                    "change in a function that calls it")
        old_calls, new_calls = _calls_to(old, callee), _calls_to(new, callee)
        for i, call in enumerate(new_calls):
            was = old_calls[i] if i < len(old_calls) else None
            was_args = [ast.dump(a) for a in was.args] if was is not None else []
            was_kw = {k.arg: ast.dump(k.value) for k in was.keywords} if was is not None else {}
            given = [(f"argument {j + 1}", a) for j, a in enumerate(call.args)
                     if j >= len(was_args) or ast.dump(a) != was_args[j]]
            given += [(f"`{k.arg}`", k.value) for k in call.keywords if was_kw.get(k.arg) != ast.dump(k.value)]
            for what, value in given:
                if _typed_number(value):
                    return (f"`{name}` passes the number `{ast.unparse(value)}` for {what} of `{callee}`: a number typed "
                            "into the call is the same typed-in value, and the study runs unchanged; pass it from what "
                            f"`{name}` is given (its parameters, the setting or the thresholds)")
    return ""


def file_blocks(reply: str) -> tuple[str, dict[str, str]]:
    """The reply's first fenced block that holds a ``def`` and is not marked for another file (``""`` when none), and
    ``{file named by its first line # file: <path>: the rest of that block}`` for the blocks so marked."""
    from .code_review import _FILE_LINE

    main, marked = "", {}
    for m in _FENCE.finditer(reply or ""):
        lines = m.group(1).strip("\n").splitlines()
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        marker = _FILE_LINE.match(lines[first]) if first is not None else None
        if marker:
            name = marker.group(1).replace("\\", "/").strip("/")
            name = name[len("code/"):] if name.startswith("code/") else name
            marked[name] = (marked[name] + "\n\n" if name in marked else "") + "\n".join(lines[first + 1:])
        elif not main and "def " in m.group(1):
            main = m.group(1)
    return main, marked


def splice(source: str, reply: str, scope: set[str] | None = None,
           ignored: list[str] | None = None) -> tuple[str | None, list[str], str]:
    """``(the file with the reply's functions put in, the names it replaced, why not)``. The first fenced block of the
    reply that holds a ``def`` is read: each function it defines that the file has AND that is in ``scope`` (the
    function(s) the finding is about) replaces it (the parameter names must be the same, in the same order; at most
    :data:`MAX_FUNCTIONS`); a definition of any other function the file already has is not spliced (its name is appended to
    ``ignored``); a function the file has not is added at the end as a helper; imports it brings are added; anything else
    in the block is ignored. ``(None, [], why)`` when nothing can be used. With no ``scope`` every function the file has
    may be replaced."""
    block = next((m.group(1) for m in _FENCE.finditer(reply or "") if "def " in m.group(1)), "")
    if not block and "def " in (reply or ""):
        block = reply
    if not block.strip():
        return None, [], "the reply held no function"
    try:
        new_tree = ast.parse(block)
    except SyntaxError as e:
        return None, [], f"the reply is not valid Python ({e.msg}, line {e.lineno})"
    try:
        old_tree = ast.parse(source)
    except SyntaxError as e:
        return None, [], f"the file does not parse ({e.msg})"
    old_funcs, new_funcs = _top_functions(old_tree), _top_functions(new_tree)
    if not new_funcs:
        return None, [], "the reply held no function"
    existing = [n for n in new_funcs if n in old_funcs]
    replaced = [n for n in existing if scope is None or n in scope]
    if ignored is not None:
        ignored.extend(n for n in existing if n not in replaced)
    if scope is not None and not replaced:
        return None, [], f"the reply does not define the function the finding is about ({', '.join(sorted(scope))})"
    if len(replaced) > MAX_FUNCTIONS:
        return None, [], f"the reply replaces {len(replaced)} functions; change only the ones the finding is about"
    for name in replaced:
        why = params_kept(old_funcs[name], new_funcs[name])
        if why:
            return None, [], f"`{name}` {why}"
    lines = source.splitlines(keepends=True)
    block_lines = block.splitlines(keepends=True)
    if block_lines and not block_lines[-1].endswith("\n"):
        block_lines[-1] += "\n"
    edits: list[tuple[int, int, str]] = []
    for name in replaced:
        old = old_funcs[name]
        start = min([old.lineno, *[d.lineno for d in old.decorator_list]])
        edits.append((start, old.end_lineno or old.lineno, _steps._segment(block_lines, new_funcs[name])))
    for first, last, text in sorted(edits, reverse=True):
        lines[first - 1:last] = [text]
    out = "".join(lines)
    added = [n for n in new_funcs if n not in old_funcs]
    if added:
        out = out.rstrip("\n") + "\n\n\n" + "\n\n".join(_steps._segment(block_lines, new_funcs[n]).rstrip("\n") + "\n" for n in added)
    have = {ast.dump(n) for n in old_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    taken: set[str] = set()
    for n in old_tree.body:
        taken |= _bound_names(n)
    imports = [_steps._segment(block_lines, n).rstrip("\n") for n in new_tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom)) and ast.dump(n) not in have and not (_bound_names(n) & taken)]
    if imports:
        out = _steps._with_imports(out, imports)
    try:
        ast.parse(out)
    except SyntaxError as e:
        return None, [], f"the file would not be valid Python ({e.msg}, line {e.lineno})"
    return out, [*replaced, *added], ""


# --- one issue's scope ---------------------------------------------------------------------------------------------------


def _def_sources(source: str) -> dict[str, str]:
    """``{qualified name: the text of that def}`` for every function and method of ``source``, nested ones included; the text
    of a def leaves out the defs inside it (they have their own entry), so a change to a nested function is a change to it
    alone."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines()
    out: dict[str, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                first = min([child.lineno, *[d.lineno for d in child.decorator_list]])
                inner = [(min([c.lineno, *[d.lineno for d in c.decorator_list]]), c.end_lineno or c.lineno)
                         for c in ast.walk(child) if c is not child and isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))]
                kept = [text for n, text in enumerate(lines[first - 1:(child.end_lineno or child.lineno)], first)
                        if not any(a <= n <= b for a, b in inner)]
                out[name] = "\n".join(kept)
                walk(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def _bound_names(node: ast.stmt) -> set[str]:
    """The names a top-level statement binds."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
    return names


def _frame(tree: ast.Module) -> list[tuple[str, str]]:
    """Every statement that is not a function, as ``(where, ast.dump)``: the module's own statements (constants, imports,
    guards) and those in the body of a class (its header and its class-level assignments); the methods and functions are
    compared apart (:func:`other_functions_changed`)."""
    out: list[tuple[str, str]] = []

    def walk(body: list[ast.stmt], where: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.ClassDef):
                bare = copy.copy(node)
                bare.body = []
                header = ast.dump(bare)
                out.append((where, header))
                walk(node.body, f"{where}{node.name}.")
            else:
                out.append((where, ast.dump(node)))

    walk(tree.body, "")
    return out


def without_main_guard(old: str, new: str) -> tuple[str, str]:
    """When ``new`` moves module-level statements of ``old`` under an ``if __name__ == "__main__":`` block ``old`` did not
    have (so importing the file no longer runs them; a module that ran its study when a test imported it), the two texts
    to compare without that move: ``old`` without the moved statements, ``new`` without the block. A statement counts as
    moved when it is in the block, or in a function ``new`` adds, as it was. Otherwise both are given back as they are."""
    from .code_layout import _is_main_guard

    try:
        before, after = ast.parse(old), ast.parse(new)
    except SyntaxError:
        return old, new
    guards = [n for n in after.body if _is_main_guard(n)]
    if any(_is_main_guard(n) for n in before.body) or len(guards) != 1:
        return old, new
    had = set(_top_functions(before))
    inside = {ast.dump(s) for s in ast.walk(guards[0]) if isinstance(s, ast.stmt)}
    for fn in after.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name not in had:
            inside |= {ast.dump(s) for s in ast.walk(fn) if isinstance(s, ast.stmt) and s is not fn}
    stays = {ast.dump(n) for n in after.body if n is not guards[0]}
    moved = [n for n in before.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and ast.dump(n) not in stays and ast.dump(n) in inside]
    if not moved:
        return old, new
    before.body = [n for n in before.body if n not in moved]
    after.body = [n for n in after.body if n is not guards[0]]
    return ast.unparse(before), ast.unparse(after)


def frame_changes(old: str, new: str) -> list[str]:
    """What a repair changed outside the functions: a module-level or class-level statement of ``old`` that is not in ``new``
    as it was, or one that ``new`` adds. Allowed additions, which are not listed: a new top-level function or class (a
    helper), and a new import that binds only names the file did not bind before (it brings a module in, and changes no
    name that exists). Everything else, a constant given another value included, is a change."""
    try:
        before, after = ast.parse(old), ast.parse(new)
    except SyntaxError:
        return ["the file does not parse"]
    taken: set[str] = set()
    for node in before.body:
        taken |= _bound_names(node)
    skip: set[str] = set()  # new classes (with their class-level statements) and new imports that change no name
    new_classes: set[str] = set()
    for node in after.body:
        if isinstance(node, ast.ClassDef) and node.name not in taken:
            new_classes.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and not (_bound_names(node) & taken):
            skip.add(ast.dump(node))
    new_frame = [(w, d) for w, d in _frame(after) if d not in skip and w.split(".")[0] not in new_classes
                 and not any(f"ClassDef(name='{c}'" in d for c in new_classes)]
    changed = []
    for item in _frame(before):
        if item in new_frame:
            new_frame.remove(item)
        else:
            changed.append(item)
    return [f"{where or 'the module'}: {text[:60]}" for where, text in changed + new_frame]


def frame_altered(old: str, new: str) -> list[tuple[str, set[str], str]]:
    """The module- and class-level statements of ``old`` that ``new`` changed or removed (additions are not listed), each
    as ``(description, the names it binds, its ast.dump)``."""
    try:
        before, after = ast.parse(old), ast.parse(new)
    except SyntaxError:
        return [("the file does not parse", set(), "")]
    kept = [d for _w, d in _frame(after)]
    names = {ast.dump(n): _bound_names(n) for tree in (before,) for n in ast.walk(tree)
             if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom))}
    out: list[tuple[str, set[str], str]] = []
    for where, dump in _frame(before):
        if dump in kept:
            kept.remove(dump)
        else:
            out.append((f"{where or 'the module'}: {dump[:60]}", names.get(dump, set()), dump))
    return out


def _quantity_keys(issue: dict[str, Any]) -> set[str]:
    """The finding's quantities, normalised as the findings themselves are (:func:`core.code_findings.norm`)."""
    raw = [issue.get("quantity"), *(issue.get("quantities") or [])]
    return {k for k in (_issues.norm(q) for q in raw) if k}


def _frame_assigns(source: str) -> list[tuple[str, str, ast.stmt]]:
    """The assignments at module or class level of ``source`` (where the statements of :func:`_frame` are), as
    ``(where, ast.dump, node)``; those inside a function or a nested block are not."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[tuple[str, str, ast.stmt]] = []

    def walk(body: list[ast.stmt], where: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, f"{where}{node.name}.")
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                out.append((where, ast.dump(node), node))

    walk(tree.body, "")
    return out


def _targets(node: ast.stmt) -> list[ast.expr]:
    return list(node.targets) if isinstance(node, ast.Assign) else [node.target]  # type: ignore[attr-defined]


def _dict_value(node: ast.stmt | None) -> ast.Dict | None:
    value = getattr(node, "value", None)
    return value if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Dict) else None


def _quantity_name(key: ast.expr | None, quantities: set[str]) -> bool:
    return isinstance(key, ast.Constant) and isinstance(key.value, str) and _issues.norm(key.value) in quantities


def changes_only_quantity_entries(old: ast.stmt, new: ast.stmt | None, quantities: set[str]) -> bool:
    """Whether a changed statement changes only the entries of the finding's own quantity: a dict literal assigned to the
    same targets in which every existing entry that changed or went is keyed by a quantity of the finding (entries of the
    quantity may be added; a ``**`` entry may not be added, taken out or changed), or
    an assignment (``=``, ``+=``, annotated) to the one subscript ``name["quantity"]``, the same on both sides. Anything
    else, and a statement that is gone, is not."""
    if not quantities or new is None or type(old) is not type(new):
        return False
    old_targets, new_targets = _targets(old), _targets(new)
    if [ast.dump(t) for t in old_targets] != [ast.dump(t) for t in new_targets]:
        return False
    if any(isinstance(t, (ast.Tuple, ast.List, ast.Starred)) for t in old_targets):
        return False
    before, after = _dict_value(old), _dict_value(new)
    if before is not None or after is not None:
        if before is None or after is None:
            return False
        old_keys = [ast.dump(k) if k is not None else None for k in before.keys]
        new_keys = [ast.dump(k) if k is not None else None for k in after.keys]
        named = [k for k in new_keys if k is not None]
        if len(named) != len(set(named)):
            return False
        # The entries the dict had keep their order (a ``**`` among them, whose place decides what it overwrites), apart
        # from entries of the quantity, which may go, come or stand where one went; a ``**`` is never added or changed,
        # and no other key is added.
        if any(isinstance(n, ast.NamedExpr) for n in ast.walk(after)):
            return False
        old_ids = [("key", k) if k is not None else ("**", ast.dump(v)) for k, v in zip(old_keys, before.values)]
        new_ids = [(("key", key) if key is not None else ("**", ast.dump(v)), k)
                   for key, k, v in zip(new_keys, after.keys, after.values)]
        kept = [i for i, k in zip(old_ids, before.keys) if i in [n for n, _k in new_ids] or not _quantity_name(k, quantities)]
        rest = [n for n, k in new_ids if n in old_ids or not _quantity_name(k, quantities)]
        if rest != kept:  # a key that is new and not the quantity's, a new or moved ``**``, an entry out of order
            return False
        now = {k: ast.dump(v) for k, v in zip(new_keys, after.values) if k is not None}
        for k, v, key in zip(before.keys, before.values, old_keys):
            if key is not None and now.get(key) != ast.dump(v) and not _quantity_name(k, quantities):
                return False
        return any(_quantity_name(k, quantities) for k in before.keys + after.keys)
    return len(old_targets) == 1 and isinstance(old_targets[0], ast.Subscript) and _quantity_name(old_targets[0].slice, quantities)


def _containers(source: str, quantities: set[str]) -> set[str]:
    """The names under which ``source`` already keeps an entry of the finding's quantity at module or class level: a
    name assigned a dict literal with a key of the quantity, or one whose ``name["quantity"]`` is assigned."""
    found: set[str] = set()
    for _w, _d, node in _frame_assigns(source):
        targets = _targets(node)
        if len(targets) != 1:
            continue
        value = _dict_value(node)
        if isinstance(targets[0], ast.Name) and value is not None and any(_quantity_name(k, quantities) for k in value.keys):
            found.add(targets[0].id)
        elif isinstance(targets[0], ast.Subscript) and isinstance(targets[0].value, ast.Name)                 and _quantity_name(targets[0].slice, quantities):
            found.add(targets[0].value.id)
    return found


def _adds_quantity_entry(node: ast.stmt, quantities: set[str], bound: set[str]) -> bool:
    """Whether an added statement is a set of the finding's quantity entry (``name["quantity"] = ...``, ``+=``,
    annotated) of a name that already holds an entry of that quantity (``bound``: :func:`_containers`). The value
    written is not looked at, apart from a walrus (it binds a name unseen): the finding's own test and the regression
    checks run it."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) or any(isinstance(n, ast.NamedExpr) for n in ast.walk(node)):
        return False
    targets = _targets(node)
    return (len(targets) == 1 and isinstance(targets[0], ast.Subscript) and isinstance(targets[0].value, ast.Name)
            and targets[0].value.id in bound and _quantity_name(targets[0].slice, quantities))


def _added(old: str, new: str) -> list[tuple[str, str, ast.stmt]]:
    """The module- and class-level statements ``new`` has that ``old`` does not, as ``(where, ast.dump, node)``, apart
    from what :func:`frame_changes` allows (a new function or class, a new import that binds only names not bound
    before)."""
    before, after = ast.parse(old), ast.parse(new)
    taken: set[str] = set()
    for node in before.body:
        taken |= _bound_names(node)
    new_classes = {n.name for n in after.body if isinstance(n, ast.ClassDef) and n.name not in taken}
    kept = _frame_nodes(before)
    out = []
    for where, dump, node in _frame_nodes(after):
        if (isinstance(node, (ast.Import, ast.ImportFrom)) and where == "" and not (_bound_names(node) & taken)
                and not any(a.name == "*" for a in node.names)):
            continue
        if where.split(".")[0] in new_classes or (isinstance(node, ast.ClassDef) and where == "" and node.name in new_classes):
            if _plain_class_statement(node):
                continue
            out.append((where, dump, node))
            continue
        match = next((k for k in kept if k[:2] == (where, dump)), None)
        if match:
            kept.remove(match)
        else:
            out.append((where, dump, node))
    return out


def _plain_class_statement(node: ast.stmt) -> bool:
    """A statement a new class may have: a nested class, a docstring, ``pass``, or an assignment to plain names (class
    attributes) with no walrus in it. (Its methods are functions, which :func:`_frame_nodes` leaves out.)"""
    if isinstance(node, (ast.ClassDef, ast.Pass)):
        return True
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    return (isinstance(node, (ast.Assign, ast.AnnAssign)) and all(isinstance(t, ast.Name) for t in _targets(node))
            and not any(isinstance(n, ast.NamedExpr) for n in ast.walk(node)))


def _frame_nodes(tree: ast.Module) -> list[tuple[str, str, ast.stmt]]:
    """:func:`_frame` with the statement node of each item."""
    out: list[tuple[str, str, ast.stmt]] = []

    def walk(body: list[ast.stmt], where: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.ClassDef):
                bare = copy.copy(node)
                bare.body = []
                out.append((where, ast.dump(bare), node))
                walk(node.body, f"{where}{node.name}.")
            else:
                out.append((where, ast.dump(node), node))

    walk(tree.body, "")
    return out


def unspoken_frame_changes(old: str, new: str, issue: dict[str, Any]) -> list[str]:
    """What a repair changed at module or class level that the finding does not speak of: a statement that binds no name
    the finding names and is not a change of the entries of the finding's own quantity
    (:func:`changes_only_quantity_entries`)."""
    spoken, quantities = issue_names(issue), _quantity_keys(issue)
    everything = frame_altered(old, new)
    if everything and not everything[0][2]:  # the file does not parse
        return [t for t, _n, _d in everything]
    olds, news = _frame_assigns(old), _added(old, new)
    out = []
    for text, names, dump in everything:
        found = next((o for o in olds if o[1] == dump), None)
        if found is not None:
            olds.remove(found)
        value = _dict_value(found[2]) if found is not None else None
        if names & spoken and not (value is not None and any(_quantity_name(k, quantities) for k in value.keys)):
            # it binds a name the finding speaks of (a dict that holds the finding's quantity is held to the entries
            # of that quantity whatever it is called): its replacement is taken out of the additions
            twin = next((n for n in news if _bound_names(n[2]) & names), None)
            if twin:
                news.remove(twin)
            continue
        twin = None
        if found is not None:
            twin = next((n for n in news if n[0] == found[0] and isinstance(n[2], (ast.Assign, ast.AnnAssign, ast.AugAssign))
                         and [ast.dump(t) for t in _targets(n[2])] == [ast.dump(t) for t in _targets(found[2])]), None)
        if twin:
            news.remove(twin)
        if found is None or not changes_only_quantity_entries(found[2], twin[2] if twin else None, quantities):
            out.append(text)
    bound = _containers(old, quantities)
    for where, dump, node in news:  # what is added and is no replacement of a statement that went
        if not _adds_quantity_entry(node, quantities, bound):
            out.append(f"{where or 'the module'}: added {dump[:60]}")
    return out


def issue_names(issue: dict[str, Any]) -> set[str]:
    """The code names a finding speaks of: its quantity and function, the parts of a dotted name, and in its wording only
    what reads as a name in code (quoted in backticks, or with an underscore or a digit in it), never a plain word such
    as "data" or "value" that the prose happens to use."""
    names: set[str] = set()
    for key in ("quantity", "function"):
        names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(issue.get(key) or "")))
    texts = [str(issue.get(k) or "") for k in ("headline", "summary", "detail", "claim")]
    texts += [str(w) for w in issue.get("wordings") or []]
    for text in texts:
        for quoted in re.findall(r"`([^`]+)`", text):
            names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", quoted))
        names |= {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text) if "_" in w.strip("_") or re.search(r"\d", w)}
    return names


def other_functions_changed(old: str, new: str, allowed: str) -> list[str]:
    """The functions, methods and nested functions other than ``allowed`` (a qualified name) whose own text is not byte for
    byte what it was, or that are gone."""
    before, after = _def_sources(old), _def_sources(new)
    return sorted(name for name, text in before.items() if name != allowed and after.get(name) != text)


def target_file(issue: dict[str, Any], code_dir: Path) -> str:
    """The file an issue is repaired in: the one it names, else by what it is about (the simulation or the analysis, as
    :func:`core.code_review.files_for` decides)."""
    from . import code_review as _review

    if issue.get("file"):
        return str(issue["file"])
    probe = {"kind": "logic" if issue.get("kind") == "logic" else str(issue.get("kinds", ["requirement"])[0]),
             "file": "", "severity": issue.get("severity", "medium"), "headline": issue.get("headline", False)}
    got = _review.files_for([probe], code_dir)
    return next(iter(got), "")


def scope_function(issue: dict[str, Any], source: str) -> str:
    """The top-level function a repair is limited to, or ``""`` when the issue names none the file has."""
    name = str(issue.get("function") or "")
    if not name or name.startswith("<") or "." in name:
        return ""
    try:
        return name if name in _top_functions(ast.parse(source)) else ""
    except SyntaxError:
        return ""


def signatures(source: str, skip: str = "") -> str:
    """One line per top-level function of ``source`` (its ``def`` line): what the rest of the file offers."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "(the file does not parse)"
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name != skip:
            out.append(f"def {n.name}({', '.join(_steps._param_names(n))})")
    return "\n".join(out) or "(none)"


def function_source(source: str, name: str) -> str:
    try:
        node = _top_functions(ast.parse(source)).get(name)
    except SyntaxError:
        return ""
    return (ast.get_source_segment(source, node) or "") if node is not None else ""


def finding_block(issue: dict[str, Any], scrub: Callable[[str], str]) -> str:
    """The issue as the requests show them: each problem of it in full (where, what, why, how to fix), with the plan's values
    taken out."""
    body = scrub(str(issue.get("statement") or "")).strip()
    if not body:
        where = issue.get("where") or issue.get("file") or "not stated"
        body = "\n".join([f"Where: {where}", f"What is wrong: {scrub(str(issue.get('text') or ''))}",
                          f"Why it is wrong against the plan: {scrub(str(issue.get('why') or '')) or 'not stated'}",
                          f"How to fix it: {scrub(str(issue.get('fix') or '')) or 'not stated'}"])
    if issue.get("quantity"):
        body += f"\nThe quantity it concerns: {issue['quantity']}"
    return body


# --- the loop ----------------------------------------------------------------------------------------------------------


@dataclass
class Env:
    """What the loop needs from the engine. ``ask(prompt, who, kind)``: one request (``who`` is ``code`` for the model
    that writes the code, ``other`` for the other model; ``kind`` is ``test`` or ``repair``). ``run_tests(paths)`` runs test
    files on the code as it is. ``snapshot()`` gives ``{check: pass/fail/error}`` for the checks that must keep passing
    (the equation tests, every accepted finding test) and ``static()`` the fault count of each static check. ``keeps(old,
    new, rel)`` is the reason a changed file may not replace the old one (``""``: it may)."""

    root: Path
    code: Path
    record: dict[str, Any]
    save: Callable[[], None]
    ask: Callable[[str, str, str], Awaitable[str]]
    run_tests: Callable[[list[str]], Awaitable[dict[str, TestRun]]]
    snapshot: Callable[[], Awaitable[dict[str, str]]]
    static: Callable[[], dict[str, int]]
    keeps: Callable[[str, str, str], str]
    prompts: dict[str, string.Template]
    topic: str
    equations: str
    scrub: Callable[[str], str]
    log: Any
    package: str = ""
    other: str = ""  # the name of the other model; empty when there is none
    code_model: str = ""
    on_accept: Callable[[str, str, str, dict[str, Any]], None] | None = None
    #: FI's small run of the simulation on the code as it is: ``""`` when it ran, else why not (a repair that breaks it is
    #: not kept); ``None`` when there is none to make.
    small_run: Callable[[], Awaitable[str]] | None = None
    changed: list[str] = field(default_factory=list)
    baseline: dict[str, str] = field(default_factory=dict)
    baseline_static: dict[str, int] = field(default_factory=dict)


def _accepted(record: dict[str, Any]) -> list[str]:
    value = record.setdefault("accepted_tests", [])
    return value if isinstance(value, list) else []


def test_path(issue: dict[str, Any]) -> str:
    n = int(issue.get("tests") or 0)
    return (TEST_DIR / f"{issue['id']}_{n}.py").as_posix()


def spend(env: Env, issue: dict[str, Any], kind: str) -> bool:
    """Count one request on disk before it is made. ``False`` when the issue's bound for ``kind`` or the quest's is spent."""
    from . import code_review as _review

    caps = {"test": TEST_CALLS, "repair": REPAIR_CALLS, "other": OTHER_CALLS}
    calls = issue.setdefault("calls", {"test": 0, "repair": 0, "other": 0})
    if int(env.record.get("finding_calls") or 0) >= _review.MAX_FINDING_CALLS or int(calls.get(kind) or 0) >= caps[kind]:
        return False
    calls[kind] = int(calls.get(kind) or 0) + 1
    env.record["finding_calls"] = int(env.record.get("finding_calls") or 0) + 1
    key = "test_calls" if kind == "test" else "repair_calls"
    env.record[key] = int(env.record.get(key) or 0) + 1
    env.save()
    return True


async def _ask(env: Env, prompt: str, who: str, kind: str) -> str:
    try:
        return await env.ask(prompt, who, kind)
    except Exception as e:  # noqa: BLE001 -- _ModelAnswerProblem is the engine's to raise: it re-raises through ``ask``
        if type(e).__name__ == "_ModelAnswerProblem":
            raise
        env.log.warning("[code_review] a request for %s could not be made: %r", kind, e)
        return ""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _test_prompt(env: Env, issue: dict[str, Any], rel: str, problem: str) -> str:
    source = _read(env.code / rel)
    scope = scope_function(issue, source)
    return env.prompts["code_finding_test"].substitute(
        topic=env.topic or "(not stated)", equations=env.equations, finding=finding_block(issue, env.scrub),
        file=rel, code=(f"### {rel} (the function `{scope}`)\n```python\n{function_source(source, scope)}\n```\n\n"
                        f"The rest of {rel} offers:\n{signatures(source, scope)}" if scope else
                        f"### {rel}\n```python\n{source.rstrip()}\n```"),
        modules=module_names(env.code, env.package), problem=problem or "(none: this is the first try)")


def _error_problem(detail: str) -> str:
    return (f"the test could not show anything: {detail}. It must FAIL BY AN ASSERTION about the values the code "
            "computes, on the current code; an exception in the test itself shows nothing.")


async def ask_again(env: Env, issue: dict[str, Any], rel: str, error: TestRun) -> TestRun | None:
    """A shown issue's test now errors in its own code (not inside the program): the code model is asked ONCE more, shown
    the error, for a test written as an assertion about the values the code computes, and it is run twice. Its
    ``TestRun``: ``fail`` (a failing assertion in both runs; the new test becomes the issue's test), ``pass``, or ``error``;
    ``None`` when it was asked before for this issue, or no test came back. What happened is kept in ``issue["asked_again"]``.
    Counted in the record's requests like every other; an error never stops anything."""
    if issue.get("asked_again"):
        return None
    issue["asked_again"] = {"error": error.detail[:300], "at": error.at, "outcome": "no test came back"}
    env.record["finding_calls"] = int(env.record.get("finding_calls") or 0) + 1
    env.record["test_calls"] = int(env.record.get("test_calls") or 0) + 1
    env.save()
    reply = await _ask(env, _test_prompt(env, issue, rel, _error_problem(error.detail)), "code", "test")
    text = reply_test(reply)
    if not text:
        env.save()
        return None
    issue["tests"] = int(issue.get("tests") or 0) + 1
    while (env.root / test_path(issue)).exists():  # the test that errored is kept as it is
        issue["tests"] += 1
    path = test_path(issue)
    try:
        (env.root / path).parent.mkdir(parents=True, exist_ok=True)
        (env.root / path).write_text(text, encoding="utf-8")
    except OSError as e:
        issue["asked_again"]["outcome"] = f"the test could not be written: {e!r}"[:200]
        env.save()
        return None
    first = (await env.run_tests([path])).get(path) or TestRun("error", "not run")
    run = first
    if first.status == "fail":
        second = (await env.run_tests([path])).get(path) or TestRun("error", "not run")
        if second.status != "fail":
            run = second
    issue["asked_again"].update(test=path, outcome=run.status, detail=run.detail[:300], at=run.at or error.at)
    if run.status == "fail":
        issue.update(test=path, failure=run.detail[:300], failure_kind=_issues.ASSERTION, assertion=run.line[:200],
                     assertion_at=run.at)
    env.save()
    return run


async def demonstrate(env: Env, issue: dict[str, Any], rel: str) -> bool:
    """Ask for a test that fails on the current code because of the finding, and run it. ``True`` when it fails by an
    assertion on two runs (the issue is demonstrated and the test kept); otherwise the issue is ``not demonstrated``."""
    detail = ""
    last = "no usable test could be made"
    while spend(env, issue, "test"):
        prompt = _test_prompt(env, issue, rel, detail)
        reply = await _ask(env, prompt, "code", "test")
        text = reply_test(reply)
        if not text:
            detail = "the reply held no Python test: answer with one fenced block that defines a function `test_...`"
            last = "the reply held no test"
            continue
        issue["tests"] = int(issue.get("tests") or 0) + 1
        path = test_path(issue)
        full = env.root / path
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(text, encoding="utf-8")
        except OSError as e:
            env.log.warning("[code_review] could not write the test for %s: %r", issue["id"], e)
            break
        first = (await env.run_tests([path])).get(path) or TestRun("error", "not run")
        if first.status == "pass":
            issue.update(demonstrated=False, outcome=_issues.NOT_DEMONSTRATED,
                         reason="the test written for this finding passes on the current code")
            env.save()
            return False
        if first.status == "fail":
            second = (await env.run_tests([path])).get(path) or TestRun("error", "not run")
            if second.status == "fail":
                # Only a test's own assertion about values shows a fault; what failed is kept, for the gate after the run.
                issue.update(demonstrated=True, test=path, failure=first.detail[:300], failure_kind=_issues.ASSERTION,
                             assertion=first.line[:200], assertion_at=first.at)
                env.save()
                return True
            detail = "the test did not fail on a second run, so it is not deterministic: fix the seeds and make it assert one thing"
            last = detail
            continue
        # An error shows nothing. Where it was raised is kept; a crash inside the code itself goes to the crash repair,
        # and a test that errors in its own code is not kept (another is asked for, within the bound).
        issue["errors"] = [*(issue.get("errors") or [])[-2:],
                           {"test": path, "kind": first.kind, "place": first.place, "at": first.at,
                            "detail": first.detail[:300]}]
        if first.crash_in_program:
            issue.update(demonstrated=False, outcome=_issues.NOT_DEMONSTRATED,
                         crash={"at": first.at, "detail": first.detail[:300], "test": path},
                         reason=f"its test crashed inside the code at {first.at or 'an unknown line'}: a crash, not a "
                                "shown fault; it goes to the crash repair")
            env.save()
            return False
        detail = _error_problem(first.detail)
        last = first.detail
    issue.update(demonstrated=False, outcome=_issues.NOT_DEMONSTRATED, reason=f"no test showed it ({last[:160]})")
    env.save()
    return False


def module_names(code: Path, package: str) -> str:
    """How a test imports the code: the scripts by file name, the package's model by its dotted name."""
    names = [f"`import {n[:-3]}`" for n in ("simulate.py", "experiment.py") if (code / n).is_file()]
    if package:
        names.append(f"`from {package} import model`")
    return ", ".join(names) or "(none)"


def _callers_block(env: Env, callers: dict[str, set[str]]) -> str:
    """The functions that call the one being repaired, shown so a repair that adds a parameter can pass it on."""
    parts = []
    for other, names in sorted(callers.items()):
        source = _read(env.code / other)
        for name in sorted(names):
            parts.append(f"### {other} (the function `{name}`, which calls it)\n```python\n{function_source(source, name)}\n```")
    return "\n\n".join(parts[:4])


async def _attempt(env: Env, issue: dict[str, Any], rel: str, who: str, detail: str) -> tuple[bool, str]:
    """One repair request and its checks. ``(accepted, why not)``; a rejected change is undone, in every file it touched.

    A repair of one function may add parameters after its own (each with a default value) and change the functions that
    call it directly, in any file of the code, to pass them on (:func:`params_kept`, :func:`direct_callers`); a repair of a
    whole file may move statements that run when the file is imported under an ``if __name__ == "__main__":`` block
    (:func:`without_main_guard`). It is accepted only when the finding's own test passes, every check that passed before
    still passes, no static check finds more faults, and FI's small run of the simulation still runs (``env.small_run``)."""
    path = env.code / rel
    old = _read(path)
    scope = scope_function(issue, old)
    test = issue.get("test") or ""
    failure = issue.get("failure") or ""
    callers = direct_callers(env.code, rel, scope) if scope else {}
    shown = _callers_block(env, callers)
    code_block = (f"### {rel} (the function `{scope}`)\n```python\n{function_source(old, scope)}\n```\n\nThe rest of {rel} "
                  f"offers (do not change these unless the finding is about them):\n{signatures(old, scope)}"
                  + (f"\n\nIt is called by:\n\n{shown}" if shown else "")
                  if scope else f"### {rel}\n```python\n{old.rstrip()}\n```")
    how = ("Reply with ONE fenced Python block holding the corrected function(s), each whole, with the same names and "
           "parameters (you may add small helper functions and import lines). If the fix needs a value the function is not "
           "given, add it as a parameter AFTER the existing ones, with a default value, and change each function that "
           "calls it (shown above) to pass it: that function whole, in its own fenced block whose first line is "
           "`# file: <its path>`." if scope else
           f"Reply with ONE fenced Python block whose first line is `# file: {rel}` and that holds the whole corrected file. "
           "If importing the file runs the study, you may put what runs it under `if __name__ == \"__main__\":`.")
    prompt = env.prompts["code_review_repair"].substitute(
        topic=env.topic or "(not stated)", equations=env.equations, finding=finding_block(issue, env.scrub),
        test=_read(env.root / test) if test else "(none)", failure=failure or "(none)", code=code_block,
        package=env.package or "(none)", how=how, problem=detail or "(none: this is the first try)")
    reply = await _ask(env, prompt, who, "repair")
    dotted = str(issue.get("function") or "")
    edits: dict[str, tuple[str, str]] = {}  # file -> (old text, new text)
    if scope:
        main, marked = file_blocks(reply)
        mine_too = callers.get(rel, set())
        skipped: list[str] = []
        new, names, why = splice(old, f"```python\n{main or marked.get(rel, '')}\n```", {scope, *mine_too}, skipped)
        if skipped:
            env.log.warning("[code_review] %s: the reply also defined %s, which the finding is not about; not used",
                            issue.get("id"), ", ".join(f"`{n}`" for n in skipped))
        if new is not None and scope not in names:
            new, why = None, f"the reply does not define the function the finding is about ({scope})"
        for other, text in marked.items():
            if new is None or other == rel:
                continue
            target = next((c for c in callers if c == other or Path(c).name == Path(other).name), None)
            if target is None:
                env.log.warning("[code_review] %s: the reply also changed %s, which does not call `%s`; not used",
                                issue.get("id"), other, scope)
                continue
            before = _read(env.code / target)
            changed, done, why_not = splice(before, f"```python\n{text}\n```", set(callers[target]), [])
            if changed is None:
                new, why = None, f"its change to {target}: {why_not}"
            elif changed != before:
                edits[target] = (before, changed)
                names = [*names, *(f"{target}:{n}" for n in done)]
    else:
        from . import code_review as _review

        new = _review.parse_files(reply, [rel]).get(rel)
        names, why = [rel], "" if new else "the reply held no whole file"
        # moving what runs on import under a main guard is not a change of the module the finding must speak of
        was, now = without_main_guard(old, new) if new else (old, new)
        if new and dotted and not dotted.startswith("<"):
            changed = other_functions_changed(old, new, dotted)
            outside = frame_changes(was, now)
            if changed:
                new, why = None, ("it changed " + ", ".join(f"`{n}`" for n in changed[:4])
                                  + f", which the finding is not about (only `{dotted}` may change)")
            elif outside:
                new, why = None, ("it changed " + "; ".join(outside[:3])
                                  + f", which the finding is not about (only `{dotted}` may change)")
        elif new:
            # A finding that names no function (a quantity the results lack, a module-level value): the repair may
            # add helpers and change ONE existing function, and may change an existing module- or class-level
            # statement only when a name it binds is one the finding speaks of, or when it is an entry of the finding's
            # own quantity (a result dict built in one module-level statement). Anything wider is not this finding's.
            changed = other_functions_changed(old, new, "")
            altered = unspoken_frame_changes(was, now, issue)
            if len(changed) > 1:
                new, why = None, ("it changed " + ", ".join(f"`{n}`" for n in changed[:4])
                                  + "; a repair of a finding that names no function may change one existing function")
            elif altered:
                new, why = None, ("it changed " + "; ".join(altered[:3])
                                  + ", which the finding does not speak of")
    if new is None or not new.strip():
        return False, why or "the reply held no usable code"
    if new != old:
        edits[rel] = (old, new)
    if not edits:
        return False, "the reply changed nothing"
    if scope:
        # a function that calls the repaired one may change only the arguments of that call
        for target, (before, after_text) in edits.items():
            problem = caller_edit_problem(before, after_text, set(callers.get(target) or ()) - {scope}, scope)
            if problem:
                return False, problem if target == rel else f"its change to {target}: {problem}"
    if scope and rel in edits:
        final = {c: (edits[c][1] if c in edits else _read(env.code / c)) for c in callers}
        unpassed = unpassed_params(function_source(old, scope), function_source(edits[rel][1], scope), scope, final,
                                   callers)
        if unpassed:
            return False, unpassed
    for target, (before, after_text) in edits.items():
        problem = env.keeps(before, after_text, target)
        if problem:
            return False, problem if target == rel else f"its change to {target}: {problem}"

    def put_back() -> None:
        for target, (before, _a) in edits.items():
            (env.code / target).write_text(before, encoding="utf-8")

    for target, (_b, after_text) in edits.items():
        (env.code / target).write_text(after_text, encoding="utf-8")
    try:
        mine = (await env.run_tests([test])).get(test) or TestRun("error", "not run")
        if mine.status != "pass":
            raise _Rejected("the finding's own test still fails: " + (mine.detail or mine.status))
        after = await env.snapshot()
        broke = regressions(env.baseline, after)
        if broke:
            raise _Rejected("it broke checks that passed before: " + ", ".join(broke[:4]))
        more = worse_static(env.baseline_static, env.static())
        if more:
            raise _Rejected("it added faults to the static checks: " + ", ".join(more))
        if env.small_run is not None:
            stopped = await env.small_run()
            if stopped:
                raise _Rejected("FI's small run of the simulation no longer runs: " + stopped)
    except _Rejected as e:
        put_back()
        return False, str(e)
    except BaseException:
        put_back()
        raise
    if env.on_accept is not None:
        for target, (before, after_text) in edits.items():
            env.on_accept(target, before, after_text, issue)
    env.baseline = {**after, f"finding:{test}": "pass"}
    env.baseline_static = env.static()
    if test not in _accepted(env.record):
        _accepted(env.record).append(test)
    for target in edits:
        if target not in env.changed:
            env.changed.append(target)
    issue["touched"] = names
    return True, ""


class _Rejected(Exception):
    pass


async def repair_issue(env: Env, issue: dict[str, Any], rel: str) -> bool:
    """Repairs of a demonstrated issue: the code model, up to its bound; then, once, the other model. ``True`` when one
    was accepted."""
    detail = ""
    plan: list[str] = ["code"] * REPAIR_CALLS + (["other"] if env.other else [])
    for who in plan:
        kind = "repair" if who == "code" else "other"
        if not spend(env, issue, kind):
            continue
        number = len(issue.setdefault("repairs", [])) + 1
        accepted, why = await _attempt(env, issue, rel, who, detail)
        by = env.other if who == "other" else env.code_model
        issue["repairs"].append({"try": number, "by": who, "model": by or who, "outcome": "accepted" if accepted else "rejected",
                                 **({} if accepted else {"why": why[:300]})})
        env.log.info("[code_review] %s: %s repair %s%s", issue["id"], who, "accepted" if accepted else "rejected",
                     "" if accepted else f" ({why[:140]})")
        if accepted:
            issue.update(outcome=_issues.FIXED, fixed_by=by or who, fixed_by_role=who)
            env.save()
            return True
        detail = why
        env.save()
    issue["outcome"] = _issues.GAVE_UP
    issue["reason"] = f"no repair passed its test and kept the earlier checks passing ({detail[:160]})" if detail else "no repair was made"
    env.save()
    return False


async def repair_issues(env: Env) -> list[str]:
    """One pass over the open issues of the latest reading, result-changing ones first. Returns the files changed."""
    from . import code_review as _review

    order = _issues.ranked(env.record)
    todo = [i for i in order if i.get("outcome") in (_issues.OPEN, _issues.REOPENED) and not i.get("plan_conflict")]
    if not todo:
        return []
    env.baseline = await env.snapshot()
    env.baseline_static = env.static()
    for issue in todo:
        if not issue.get("changes_result") and int(env.record.get("finding_calls") or 0) >= MINOR_LIMIT:
            issue["note"] = "changes nothing the study reports: recorded, not repaired"
            continue
        if int(env.record.get("finding_calls") or 0) >= _review.MAX_FINDING_CALLS:
            issue["note"] = "the requests allowed for repairing findings were spent before this one"
            continue
        rel = target_file(issue, env.code)
        if not rel or not (env.code / rel).is_file():
            issue.update(demonstrated=False, outcome=_issues.NOT_DEMONSTRATED, reason="the file it concerns is not in the code")
            continue
        if not _issues.shown_by_assertion(issue):
            if not await demonstrate(env, issue, rel):
                continue
        await repair_issue(env, issue, rel)
        env.save()
    return list(env.changed)
