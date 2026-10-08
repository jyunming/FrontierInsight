"""The model's functions, written and checked one at a time.

A quest whose code is laid out as a small research tool (:mod:`core.code_layout`) has a package ``code/<package>/`` that
holds the model's equations, one function each. Written whole, in one reply together with the scenario, a function that
is wrong is found only after the whole study is set up to run, and repaired by rewriting everything. Here, after the
outline step has named one function per equation (``functions[].implements``), the bodies are filled one function at a
time, in dependency order (``depends_on``, else the outline's order):

1. **fill** the function (one model call; the prompt holds the one equation, the function's signature and the
   signatures of the functions it depends on, not the study);
2. **import-check** it: the file parses, the signature is the outline's, and an import in the quest's own environment
   finds the function;
3. run the **equation test** (:mod:`core.equation_tests`) that covers it, when the plan gave its worked example;
4. only then go on to the next function.

A function that fails is **repaired alone**: the repair is shown that function, its signature, its equation and the
error (for a failed equation test: the function and the equation, never the expected value), not the script. At most
``repairs`` repairs per function; a function that still fails ends the step, and the code is then written whole, as it
always was (:func:`FunctionFiller.run` returns ``fell_back``), after which the equation tests before the run, the
whole-script repair and the honest stop of the existing flow apply.

The most the step can ask of the model in a quest is ``max_calls`` (:data:`MAX_CALLS`): at
most ``functions x (1 + repairs)`` per pass, and never more than that in all passes together. Every call is counted in
``.fi/function_steps.json``, which also keeps each function's status, so a resume does not fill a function that is done.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import equation_tests as _eqt

RECORD = Path(".fi") / "function_steps.json"
#: Fixed limits (not settings): repairs of ONE function before the code is written whole; model functions above which the
#: code is written whole; the most requests the step (fills, repairs, and the repairs after the equation tests) may make
#: in a whole quest.
REPAIRS = 2
MAX_FUNCTIONS = 10
MAX_CALLS = 40
#: A fenced Python block in a reply.
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
PENDING, DONE, FAILED = "pending", "done", "failed"
_STUB_MARK = "not written yet"


@dataclass
class Spec:
    name: str
    signature: str
    purpose: str
    equation: str
    depends_on: list[str] = field(default_factory=list)

    def params(self) -> list[str]:
        node = ast.parse(self.signature.rstrip().rstrip(":") + ":\n    pass").body[0]
        return _param_names(node)  # type: ignore[arg-type]


def _param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = node.args
    names = [x.arg for x in [*a.posonlyargs, *a.args]]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return names


# --- what the outline says the model's functions are -----------------------------------------------------------------


def model_functions(outline: dict[str, Any] | None, protocol: dict[str, Any] | None) -> tuple[list[Spec], list[str]]:
    """``(specs in the order they are written, why not)``. The step is used only when EVERY equation the simulation
    generates its data with (``role: generates``) has a function in the outline (``implements``: its id), each with a
    signature that parses; otherwise no specs and the reason, and the code is written whole."""
    equations = {_eqt._text(e["id"]).upper(): _eqt._text(e["id"]) for e in _eqt.eligible(protocol)}
    notes: list[str] = []
    specs: dict[str, Spec] = {}
    seen_eq: set[str] = set()
    for item in (outline or {}).get("functions") or []:
        if not isinstance(item, dict) or not _eqt._text(item.get("implements")):
            continue
        eid = equations.get(_eqt._text(item["implements"]).upper())
        name = _eqt._text(item.get("name"))
        if eid is None:
            notes.append(f"the outline's function {name or '?'} implements {item['implements']}, which the plan's model does not list")
            continue
        if eid in seen_eq:
            notes.append(f"the outline names two functions for equation {eid}")
            continue
        sig = _eqt._text(item.get("signature"))
        if not name.isidentifier() or not sig.startswith("def ") or not _signature_ok(sig, name):
            notes.append(f"the outline's signature for equation {eid} cannot be read ({sig[:80] or 'none'})")
            continue
        deps = [_eqt._text(d) for d in (item.get("depends_on") if isinstance(item.get("depends_on"), list) else [])]
        specs[name] = Spec(name=name, signature=sig, purpose=_eqt._text(item.get("one_line_purpose")) or name,
                           equation=eid, depends_on=[d for d in deps if d])
        seen_eq.add(eid)
    from . import oracle_check as _oracle

    missing = [i for i in _oracle.generating_equations(protocol) if _eqt._text(i) not in seen_eq]
    if missing:
        return [], [*notes, f"the outline names no function for equation {', '.join(missing)}"]
    if not specs:
        return [], notes or ["the outline names no function for an equation (`implements`)"]
    return _in_order(list(specs.values())), notes


def _signature_ok(sig: str, name: str) -> bool:
    try:
        node = ast.parse(sig.rstrip().rstrip(":") + ":\n    pass").body[0]
    except SyntaxError:
        return False
    return isinstance(node, ast.FunctionDef) and node.name == name


def _in_order(specs: list[Spec]) -> list[Spec]:
    """Dependencies first (``depends_on``, among these functions); the outline's order otherwise, and where FI cannot
    tell (a cycle) the outline's order for what is left."""
    by_name = {s.name: s for s in specs}
    done: list[Spec] = []
    placed: set[str] = set()
    remaining = list(specs)
    while remaining:
        ready = [s for s in remaining if all(d in placed or d not in by_name or d == s.name for d in s.depends_on)]
        step = ready[0] if ready else remaining[0]
        done.append(step)
        placed.add(step.name)
        remaining.remove(step)
    return done


def spec_hash(specs: list[Spec], protocol: dict[str, Any] | None) -> str:
    """What the filled functions were written for: the functions asked for and the equations (formula and worked
    example) they implement. A different one means the functions are written again."""
    wanted = {s.equation for s in specs}
    eqs = [{k: e.get(k) for k in ("id", "formula", "example")} for e in _eqt.eligible(protocol)
           if _eqt._text(e["id"]) in wanted]
    blob = json.dumps({"f": [[s.name, s.signature, s.equation, s.depends_on] for s in specs], "e": eqs},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --- the module file --------------------------------------------------------------------------------------------------


def stub(spec: Spec) -> str:
    """A placeholder for a function not written yet. It is ALWAYS importable: the header keeps the parameter names (and
    the defaults that are plain values) but no annotation and no default that names something (``np.ndarray``,
    ``np.float64``), because those are evaluated when the module is imported, and a name the module does not have yet
    would make the import check of every other function fail."""
    node = ast.parse(spec.signature.rstrip().rstrip(":") + ":\n    pass").body[0]
    assert isinstance(node, ast.FunctionDef)
    args = node.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if arg is not None:
            arg.annotation = None
    node.returns = None

    def plain(default: ast.expr | None) -> ast.expr | None:
        if default is None:
            return None
        try:
            ast.literal_eval(default)
            return default
        except (ValueError, SyntaxError, TypeError):
            return ast.Constant(value=None)

    args.defaults = [plain(d) for d in args.defaults]  # type: ignore[misc]
    args.kw_defaults = [plain(d) for d in args.kw_defaults]
    node.decorator_list = []
    header = ast.unparse(node).splitlines()[0]
    return (f"{header}\n    \"\"\"{spec.purpose}\"\"\"\n    # {spec.equation}\n"
            f"    raise NotImplementedError(\"{_STUB_MARK}\")\n")


def initial_module(specs: list[Spec], summary: str = "") -> str:
    head = ('"""' + (summary or "The model's equations, one function each; no scenario values.")
            + '"""\n\nfrom __future__ import annotations\n\n')
    return head + "\n\n".join(stub(s) for s in specs) + "\n"


def is_stub(source: str, name: str) -> bool:
    node = _find(source, name)
    if node is None:
        return True
    return any(isinstance(n, ast.Raise) and _STUB_MARK in ast.dump(n) for n in ast.walk(node))


def _find(source: str, name: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    return next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def reply_code(text: str) -> str:
    """The Python of a reply: the first fenced block that holds a function, else the reply itself when it parses."""
    for m in _FENCE.finditer(text or ""):
        if "def " in m.group(1):
            return m.group(1)
    body = (text or "").strip()
    try:
        ast.parse(body)
        return body
    except (SyntaxError, ValueError):
        return ""


def _segment(lines: list[str], node: ast.AST) -> str:
    start = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
    return "".join(lines[start - 1: node.end_lineno])


def replace_function(source: str, name: str, reply: str, *, signature: str, equation: str) -> tuple[str | None, str]:
    """``(new module text, "")`` with the function ``name`` of ``source`` replaced by the one in ``reply`` (a block of
    Python), or ``(None, why not)``. The reply's imports are added, and so are its helper functions and plain constants
    (a name the module does not have); anything else in it is dropped. The reply's function must have the outline's
    parameters, exactly; and the equation's label (``# E1``) is kept on it."""
    if not reply.strip():
        return None, "the reply held no code"
    try:
        new_tree = ast.parse(reply)
    except SyntaxError as e:
        return None, f"the reply is not valid Python ({e.msg}, line {e.lineno})"
    node = next((n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        return None, f"the reply holds no function named `{name}`"
    want = _param_names(ast.parse(signature.rstrip().rstrip(":") + ":\n    pass").body[0])  # type: ignore[arg-type]
    if _param_names(node) != want:
        return None, ("the function's parameters must be exactly (" + ", ".join(want) + "), as the outline fixed them; "
                      "the reply has (" + ", ".join(_param_names(node)) + ")")
    reply_lines = reply.splitlines(keepends=True)
    if reply_lines and not reply_lines[-1].endswith("\n"):
        reply_lines[-1] += "\n"
    body = _segment(reply_lines, node)
    from . import oracle_check as _oracle

    if not _oracle._labelled_in(equation, body):
        body = f"# {equation}\n{body}"  # the label: a comment on the line above the function names it
    old_tree = ast.parse(source)
    old = next((n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    src_lines = source.splitlines(keepends=True)
    taken = {getattr(n, "name", None) for n in old_tree.body} | {t.id for n in old_tree.body if isinstance(n, ast.Assign)
                                                                for t in n.targets if isinstance(t, ast.Name)}
    have_imports = {ast.dump(n) for n in old_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    imports, helpers = [], []
    for n in new_tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if ast.dump(n) not in have_imports:
                imports.append(_segment(reply_lines, n).rstrip("\n"))
        elif isinstance(n, ast.FunctionDef) and n is not node and n.name not in taken:
            helpers.append(_segment(reply_lines, n))
            taken.add(n.name)
        elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets):
            try:
                ast.literal_eval(n.value)
            except (ValueError, SyntaxError):
                continue
            if not any(t.id in taken for t in n.targets):  # type: ignore[attr-defined]
                helpers.append(_segment(reply_lines, n))
                taken.update(t.id for t in n.targets)  # type: ignore[attr-defined]
    if old is None:
        out = source.rstrip("\n") + "\n\n\n" + body
    else:
        start = min([old.lineno, *[d.lineno for d in old.decorator_list]])
        out = "".join(src_lines[: start - 1]) + body + "".join(src_lines[old.end_lineno:])
    if helpers:
        out = out.rstrip("\n") + "\n\n\n" + "\n\n".join(h.rstrip("\n") + "\n" for h in helpers)
    if imports:
        out = _with_imports(out, imports)
    try:
        ast.parse(out)
    except SyntaxError as e:
        return None, f"the module would not be valid Python ({e.msg}, line {e.lineno})"
    return out, ""


def _with_imports(source: str, imports: list[str]) -> str:
    """``imports`` added after the module's docstring and its existing imports."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    after, has_imports = 0, False
    for n in tree.body:
        is_doc = isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str)
        if is_doc or isinstance(n, (ast.Import, ast.ImportFrom)):
            after = n.end_lineno or after
            has_imports = has_imports or not is_doc
        else:
            break
    head = "".join(lines[:after]).rstrip("\n")
    tail = "".join(lines[after:]).lstrip("\n")
    gap = "\n" if has_imports or not head else "\n\n"
    return head + gap + "\n".join(imports) + "\n\n\n" + tail


# --- the checks ---------------------------------------------------------------------------------------------------------

_IMPORT_CHECK = """
import importlib, inspect, json, sys, traceback
sys.path.insert(0, {code!r})
try:
    module = importlib.import_module({module!r})
    fn = getattr(module, {name!r})
    if not callable(fn):
        raise TypeError({name!r} + " is not a function")
    print("IMPORT_CHECK: ok")
except BaseException:
    traceback.print_exc()
    sys.exit(3)
"""


def import_script(code_dir: Path | str, module: str, name: str) -> str:
    """The check as a ``-c`` script; ``code_dir`` is relative to the folder it runs from (the quest folder)."""
    return _IMPORT_CHECK.format(code=str(code_dir), module=module, name=name)


# A third-party module that is not installed is a problem of the environment, not of the function.
_MISSING_MODULE = re.compile(r"(?:ModuleNotFoundError|ImportError): No module named '([\w.]+)'")


def environment_problem(stderr: str, package: str) -> str:
    """The name of a module the environment lacks (not the model's own package), or ``""``."""
    m = _MISSING_MODULE.search(stderr or "")
    if m and m.group(1).split(".")[0] != package:
        return m.group(1)
    return ""


def exception_line(text: str, limit: int = 300) -> str:
    """The last non-empty line of an error text: the exception's type and message (the END of a traceback)."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return (lines[-1] if lines else "no message")[-limit:]


def traceback_end(text: str, limit: int = 1500) -> str:
    """At most ``limit`` characters of an error text, the END of it: a traceback ends with the exception (its type and
    message) and the last frames, which is what a reader needs; its start is the frames of the machinery that called the
    code. Cut at a line boundary, with ``...`` where the start was left out. The one helper for shortening a traceback."""
    clean = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(clean) <= limit:
        return clean
    tail = clean[-limit:]
    cut = tail.find("\n")
    if 0 <= cut < len(tail) - 1:
        tail = tail[cut + 1:]
    return "...\n" + tail


# Where an import failure is, by the traceback of the import check.
_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
MODULE_LEVEL = "<module>"
ELSEWHERE = "<elsewhere>"


def _dumps(source: str) -> set[str]:
    """One fingerprint per top-level statement of ``source`` (empty when it does not parse)."""
    try:
        return {ast.dump(n) for n in ast.parse(source).body}
    except (SyntaxError, ValueError):
        return set()


def _span(node: ast.stmt) -> tuple[int, int]:
    return min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]]), node.end_lineno or node.lineno


def _text_spans(text: str) -> list[tuple[int, int, str]]:
    """``(first line, last line, name)`` of each top-level function and each other top-level statement (``name`` is
    ``MODULE_LEVEL``) of a module that may not parse; read from the lines, not from the syntax tree."""
    spans: list[list[Any]] = []
    decorated: int | None = None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line[0] in " \t#)]}":
            continue
        if line.startswith("@"):
            decorated = decorated or number
            continue
        found = re.match(r"(?:async\s+)?def\s+(\w+)\s*\(", line)
        spans.append([decorated or number, 0, found.group(1) if found else MODULE_LEVEL])
        decorated = None
    for i, span in enumerate(spans):
        span[1] = (spans[i + 1][0] - 1) if i + 1 < len(spans) else len(text.splitlines())
    return [(a, b, n) for a, b, n in spans]


def who_at_line(source: str, line: int, name: str, known: set[str]) -> str:
    """Whose line ``line`` of the module ``source`` is, for the function ``name`` being written: ``name`` for its own
    function and for everything this attempt added to the module (a statement that was not in the module before:
    ``known`` holds the fingerprints of what was), another function's name for that function, ``MODULE_LEVEL`` for any
    other statement outside a function."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        for first, last, who in _text_spans(source):
            if first <= line <= last:
                return who
        return MODULE_LEVEL
    for node in tree.body:
        first, last = _span(node)
        if first <= line <= last:
            if getattr(node, "name", None) == name or ast.dump(node) not in known:
                return name
            return node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else MODULE_LEVEL
    return MODULE_LEVEL


def locate_failure(stderr: str, source: str, package: str, name: str, known: set[str]) -> tuple[str, int]:
    """``(who, line)`` for a failed import of ``<package>.model`` while ``name`` is being written: where the error is.
    ``who`` is ``name`` (its own function, or what this attempt added), another function's name, ``MODULE_LEVEL`` (a
    statement outside any function that was there before) or ``ELSEWHERE`` (the traceback does not touch the module:
    another file, the environment). The innermost frame inside the module's file decides."""
    suffix = f"{package}/model.py"
    lines = [int(n) for path, n in _FRAME.findall(stderr or "") if path.replace("\\", "/").endswith(suffix)]
    if not lines:
        if re.search(rf"has no attribute '{re.escape(name)}'|'{re.escape(name)}' is not a function", stderr or ""):
            return name, 0  # the module imports, but the function is not in it
        return ELSEWHERE, 0
    return who_at_line(source, lines[-1], name, known), lines[-1]


def place(who: str, line: int) -> str:
    """Where an error is, in a plain phrase."""
    at = f" (line {line})" if line else ""
    if who == ELSEWHERE:
        return "outside the model's module (another file, or the environment)"
    if who == MODULE_LEVEL:
        return f"in a statement of the module that is not in a function{at}"
    return f"in the function `{who}`{at}"


# --- the record -------------------------------------------------------------------------------------------------------


def load(quest_root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(quest_root: Path, record: dict[str, Any]) -> None:
    path = Path(quest_root) / RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    except OSError:
        pass


def is_ready(quest_root: Path, package: str) -> bool:
    """Whether the package on disk is the one the function steps wrote and checked (every function done)."""
    record = load(quest_root)
    if record.get("package") != package or record.get("status") != "done":
        return False
    return (Path(quest_root) / "code" / package / "model.py").is_file()


def spend(quest_root: Path, max_calls: int) -> bool:
    """Count one more request of the step in the quest's total (``.fi/function_steps.json``); ``False`` when the most
    the step may make in the quest (:data:`MAX_CALLS`) is spent."""
    record = load(quest_root)
    if int(record.get("calls_total") or 0) >= int(max_calls):
        return False
    record["calls_total"] = int(record.get("calls_total") or 0) + 1
    save(quest_root, record)
    return True


# --- the step ---------------------------------------------------------------------------------------------------------

ChatFn = Callable[[str], Awaitable[str]]
RunFn = Callable[[list[str]], Awaitable[tuple[int, str, str]]]


@dataclass
class Outcome:
    status: str  # done | fell_back | skipped
    calls: int = 0
    reason: str = ""
    failed: str = ""


def _text_block(items: list[str]) -> str:
    return "\n".join(items) if items else "(none)"


class FunctionFiller:
    """Fills ``specs`` into ``code/<package>/model.py`` one at a time. ``chat(prompt)`` asks the model; ``run(argv)``
    runs a command in the quest's environment from ``code/`` and returns ``(returncode, stdout, stderr)``; ``log`` is
    a logger. Nothing here knows about the engine."""

    def __init__(self, *, quest_root: Path, package: str, protocol: dict[str, Any] | None, summary: str,
                 prompts: dict[str, string.Template], chat: ChatFn, run: RunFn, log: Any, repairs: int = 2,
                 max_calls: int = 30, constants: str = "") -> None:
        self.root, self.package, self.protocol, self.summary = Path(quest_root), package, protocol, summary
        self.prompts, self.chat, self.run, self.log = prompts, chat, run, log
        self.repairs, self.max_calls, self.constants = max(0, int(repairs)), int(max_calls), constants
        self.code = self.root / "code"
        self.model_path = self.code / package / "model.py"

    # -- bookkeeping --

    def _spend(self, record: dict[str, Any]) -> bool:
        if int(record.get("calls_total") or 0) >= self.max_calls:
            return False
        record["calls_total"] = int(record.get("calls_total") or 0) + 1
        record["calls"] = int(record.get("calls") or 0) + 1
        save(self.root, record)
        return True

    def _equation(self, eid: str) -> dict[str, Any]:
        return next((e for e in _eqt.eligible(self.protocol) if _eqt._text(e["id"]) == eid), {"id": eid, "formula": ""})

    def _equation_block(self, eid: str) -> str:
        e = self._equation(eid)
        lines = [f"{eid}: {_eqt._text(e.get('formula'))}"]
        if _eqt._text(e.get("source")):
            lines.append(f"source: {_eqt._text(e.get('source'))}")
        if _eqt._text(e.get("derivation")):
            lines.append(f"derivation: {_eqt._text(e.get('derivation'))}")
        return "\n".join(lines)

    # -- one function --

    async def _check(self, spec: Spec, record: dict[str, Any], known: set[str] | None = None) -> tuple[str, str, str]:
        """``(problem, who, trace)``. ``problem`` is ``""`` when the function passes its checks, else what is wrong (never
        the expected value). ``who`` says whose fault it is: ``spec.name`` (the function being written, or what this
        attempt added to the module), another function's name, ``MODULE_LEVEL`` or ``ELSEWHERE``. ``trace`` is the end of
        the traceback when the import failed. ``known``: the fingerprints of the module's statements before this attempt."""
        known = known if known is not None else set()
        text = self.model_path.read_text(encoding="utf-8")
        try:
            ast.parse(text)
        except SyntaxError as e:
            who = who_at_line(text, e.lineno or 0, spec.name, known)
            return (f"model.py is not valid Python ({e.msg}, line {e.lineno}, {place(who, 0)})", who,
                    f"SyntaxError: {e.msg} (line {e.lineno})")
        module = f"{self.package}.model"
        rc, _out, err = await self.run(["-c", import_script("code", module, spec.name)])
        if rc != 0:
            missing = environment_problem(err, self.package)
            if missing:
                self.log.info("[implement] the import check of %s is skipped: this environment lacks %s", spec.name, missing)
            else:
                who, line = locate_failure(err, text, self.package, spec.name, known)
                end = traceback_end(err, 1500)
                if who == spec.name:
                    return "importing it failed" + (f" (line {line})" if line else "") + ":\n" + end, who, end
                return (f"importing the module failed {place(who, line)}, not in `{spec.name}`:\n" + end), who, end
        rows = _eqt.example_rows(self.protocol)
        row = next((r for r in rows if r["id"] == spec.equation), None)
        if row is None or row["state"] != "ok":
            record.setdefault("untested", {})[spec.equation] = (row or {}).get("why") or "no worked example"
            return "", "", ""
        case_list = _eqt.cases([row], {spec.equation: {"module": module, "function": spec.name, "file": f"{self.package}/model.py"}})
        _eqt.write(self.code, case_list)
        rc, out, err = await self.run(["code/" + _eqt.TEST_PATH, "--json", "--only", spec.equation])
        results = _eqt.parse_results(out)
        if results is None:
            return "the equation test did not run:\n" + traceback_end(err or out or "", 800), spec.name, ""
        bad = next((r for r in results if r.get("status") != _eqt.OK), None)
        return (_eqt.failure_message(bad, row["formula"]), spec.name, "") if bad else ("", "", "")

    def _drop_added_import(self, trace: str, own: set[str]) -> str:
        """An import line this function's own attempt added to the module, and at which the import of the module failed,
        is taken out (a repair adds imports but never removes one, so it could not mend it): the line, or ``""``."""
        lines = [int(n) for path, n in _FRAME.findall(trace or "") if path.replace("\\", "/").endswith(f"{self.package}/model.py")]
        if not lines:
            return ""
        text = self.model_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return ""
        for node in tree.body:
            first, last = _span(node)
            if first <= lines[-1] <= last:
                if isinstance(node, (ast.Import, ast.ImportFrom)) and ast.dump(node) in own:
                    kept = text.splitlines(keepends=True)
                    self.model_path.write_text("".join(kept[: first - 1] + kept[last:]), encoding="utf-8")
                    return ast.unparse(node)
                break
        return ""

    async def _fix_elsewhere(self, who: str, trace: str, specs: list[Spec], record: dict[str, Any],
                             known: set[str]) -> bool:
        """The import of the module fails inside ANOTHER function of the package (``who``): fix that, without spending the
        repairs of the function being written. First (when the module parses and the function is written) one repair of
        that function alone, with a plain directive; then, and when the module does not parse or the repair is unusable,
        the function goes back to its stub, which always imports, and is written again later. At most twice per function.
        ``False`` when nothing can be done (a statement outside any function, a cause outside the module)."""
        other = next((s for s in specs if s.name == who), None)
        fixes = record.setdefault("elsewhere", {})
        if other is None or int(fixes.get(who, 0)) >= 2:
            return False
        fixes[who] = int(fixes.get(who, 0)) + 1
        source = self.model_path.read_text(encoding="utf-8")
        node = _find(source, who)
        if fixes[who] == 1 and node is not None and not is_stub(source, who) and self._spend(record):
            directive = (f"Importing the module `{self.package}.model` stops inside this function, so no function of the "
                         "package can be used until it is fixed. The error ends:\n" + trace)
            try:
                new, _why = await repair_one(
                    source=source, name=who, signature=other.signature, equation=other.equation,
                    equation_block=self._equation_block(other.equation), problem=directive,
                    prompt=self.prompts["repair"], chat=self.chat, package=self.package)
            except Exception as e:  # noqa: BLE001 -- a repair that could not be asked for leaves the module as it was
                new = None
                self.log.warning("[implement] the repair of %s could not be asked for: %r", who, e)
            if new is not None:
                self.model_path.write_text(new, encoding="utf-8")
                fixed = _find(new, who)
                if fixed is not None:
                    known.add(ast.dump(fixed))
                still, _who, _trace = await self._check(other, record, known)
                record["functions"][who]["status"] = PENDING if still else DONE  # passes: no need to write it again
                self.log.info("[implement] importing the module failed inside %s, not in the function being written; "
                              "%s was repaired alone", who, who)
                return True
        text = source
        stubbed = stub(other)
        if node is not None:
            first, last = _span(node)
            lines = source.splitlines(keepends=True)
            text = "".join(lines[: first - 1]) + stubbed + "\n\n" + "".join(lines[last:])
        else:
            for first, last, name in _text_spans(source):
                if name == who:
                    lines = source.splitlines(keepends=True)
                    text = "".join(lines[: first - 1]) + stubbed + "\n\n" + "".join(lines[last:])
                    break
            else:
                return False
        self.model_path.write_text(text, encoding="utf-8")
        again = _find(text, who)
        if again is not None:
            known.add(ast.dump(again))
        record["functions"][who]["status"] = PENDING
        self.log.info("[implement] importing the module failed inside %s, not in the function being written; "
                      "%s goes back to its placeholder and is written again", who, who)
        return True

    async def _fill(self, spec: Spec, record: dict[str, Any], specs: list[Spec]) -> str:
        """Fill ``spec`` and repair it alone until it passes or its repairs are spent. ``""`` when it passes, else why not.
        A failure that is not this function's (the import of the module fails inside another function, or in a statement
        outside any function) spends none of its repairs: the cause is fixed first (:meth:`_fix_elsewhere`), and when it
        cannot be, the step ends at once naming where the error is (``record["blamed"]``)."""
        entry = record["functions"][spec.name]
        source = self.model_path.read_text(encoding="utf-8")
        related = [f"{s.signature}   # {s.equation}: {s.purpose}" for s in specs
                   if s.name in spec.depends_on and s.name != spec.name]
        problem = ""
        own: set[str] = set()
        for attempt in range(1 + self.repairs):
            if not self._spend(record):
                return "the most requests the step may make in this quest are spent"
            entry["attempts"] = int(entry.get("attempts") or 0) + 1
            if attempt == 0:
                prompt = self.prompts["fill"].substitute(
                    model_block=self.summary or "(not stated)", equation_block=self._equation_block(spec.equation),
                    function_block=f"{spec.signature}\n    # {spec.purpose}", related_block=_text_block(related),
                    constants_block=self.constants or "(none)", package=self.package)
            else:
                current = _find(source, spec.name)
                prompt = self.prompts["repair"].substitute(
                    equation_block=self._equation_block(spec.equation),
                    function_block=ast.get_source_segment(source, current) if current is not None else spec.signature,
                    problem=problem, related_block=_text_block(related), package=self.package)
            try:
                reply = await self.chat(prompt)
            except Exception as e:  # noqa: BLE001 -- a call that failed is a failed attempt
                problem = f"the request to the model failed ({type(e).__name__}: {str(e)[:120]})"
                continue
            new, why = replace_function(source, spec.name, reply_code(reply), signature=spec.signature,
                                        equation=spec.equation)
            if new is None:
                problem = why
                continue
            # The module's statements before this attempt, without what this function itself added in earlier attempts: what
            # is not among them is this function's (a statement it added, a helper it wrote).
            known = _dumps(source) - own
            own |= _dumps(new) - _dumps(source)
            self.model_path.write_text(new, encoding="utf-8")
            problem, who, trace = await self._check(spec, record, known)
            while problem and who and who != spec.name:
                if not await self._fix_elsewhere(who, trace, specs, record, known):
                    break
                problem, who, trace = await self._check(spec, record, known)
            if not problem:
                return ""
            if who and who != spec.name:
                record["blamed"] = who
                return problem
            dropped = self._drop_added_import(trace, own)
            if dropped:
                problem += f"\n(The line `{dropped}` was taken out of the module, because importing it failed.)"
            source = self.model_path.read_text(encoding="utf-8")
        return problem

    async def run_all(self, specs: list[Spec]) -> Outcome:
        """Write every function of ``specs`` in order. Returns ``done`` when all pass, ``fell_back`` when one ends the
        step (its files are removed, the code is then written whole), ``skipped`` when it was not used."""
        digest = spec_hash(specs, self.protocol)
        record = load(self.root)
        if record.get("package") != self.package or record.get("hash") != digest:
            record = {"calls_total": int(record.get("calls_total") or 0), "package": self.package, "hash": digest,
                      "functions": {}, "calls": 0, "status": "running"}
        elif record.get("status") == "fell_back":
            return Outcome("skipped", reason=str(record.get("reason") or "an earlier pass of the step fell back"))
        record["functions"] = {s.name: record["functions"].get(s.name) or
                               {"equation": s.equation, "status": PENDING, "attempts": 0} for s in specs}
        pkg_dir = self.code / self.package
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text(
            '"""' + (self.summary.splitlines()[0][:100] if self.summary else "The model's equations")
            + '. One function per equation; no scenario values."""\n\nfrom . import model  # noqa: F401\n',
            encoding="utf-8")
        if not self.model_path.is_file():
            for entry in record["functions"].values():
                entry["status"] = PENDING  # the file is gone: what the record calls done is not on disk
            self.model_path.write_text(initial_module(specs, self.summary), encoding="utf-8")
        else:
            on_disk = self.model_path.read_text(encoding="utf-8")
            if not _dumps(on_disk):  # a module that does not parse is written again from the placeholders
                for entry in record["functions"].values():
                    entry["status"] = PENDING
                self.model_path.write_text(initial_module(specs, self.summary), encoding="utf-8")
                on_disk = self.model_path.read_text(encoding="utf-8")
            for s in specs:
                if record["functions"][s.name]["status"] == DONE and is_stub(on_disk, s.name):
                    record["functions"][s.name]["status"] = PENDING
                if _find(on_disk, s.name) is None:
                    self.model_path.write_text(on_disk.rstrip("\n") + "\n\n\n" + stub(s), encoding="utf-8")
                    on_disk = self.model_path.read_text(encoding="utf-8")
        save(self.root, record)
        for _pass in range(3):  # a function sent back to its placeholder while another was written is written in the next pass
            for spec in specs:
                entry = record["functions"][spec.name]
                if entry["status"] == DONE:
                    self.log.info("[implement] function %s (equation %s) is already written and checked", spec.name, spec.equation)
                    continue
                self.log.info("[implement] writing function %s for equation %s", spec.name, spec.equation)
                problem = await self._fill(spec, record, specs)
                if problem:
                    return self._give_up(record, spec, problem)
                entry.update(status=DONE, error="")
                save(self.root, record)
                self.log.info("[implement] function %s passes its checks%s", spec.name,
                              "" if spec.equation not in (record.get("untested") or {}) else
                              f" (equation {spec.equation} has no worked example to test it with)")
            if all(record["functions"][s.name]["status"] == DONE for s in specs):
                break
        else:
            left = next(s for s in specs if record["functions"][s.name]["status"] != DONE)
            return self._give_up(record, left, "it could not be written again after the module's import failed inside another function")
        record["status"] = "done"
        save(self.root, record)
        return Outcome("done", int(record.get("calls") or 0))

    def _give_up(self, record: dict[str, Any], spec: Spec, problem: str) -> Outcome:
        """The step ends: the error is recorded where it is (the END of the traceback, and whose it is), the half-written
        package is removed and the code is written whole."""
        blamed = str(record.pop("blamed", "") or "")
        target = spec
        if blamed in record["functions"] and blamed != spec.name:
            target = Spec(name=blamed, signature="", purpose="", equation=str(record["functions"][blamed].get("equation")))
        entry = record["functions"][target.name]
        entry.update(status=FAILED, error=traceback_end(problem, 800))
        if blamed and blamed != spec.name:
            where = (f"the import of the module failed inside function {blamed}" if blamed in record["functions"]
                     else f"the import of the module failed {place(blamed, 0)}")
            record.update(status="fell_back", reason=f"function {spec.name} (equation {spec.equation}) could not be checked: "
                                                     f"{where}, not in {spec.name}")
        else:
            record.update(status="fell_back", reason=f"function {spec.name} (equation {spec.equation}) did not pass its checks")
        save(self.root, record)
        self.log.warning("[implement] function %s for equation %s still fails after %d repair(s) (%s ... %s); "
                         "the code is written whole instead", spec.name, spec.equation, self.repairs,
                         (problem.splitlines() or [""])[0][:120], exception_line(problem, 200))
        self._remove()
        return Outcome("fell_back", int(record.get("calls") or 0), record["reason"], target.name)

    def _remove(self) -> None:
        import shutil

        shutil.rmtree(self.code / self.package, ignore_errors=True)
        try:
            (self.code / _eqt.TEST_PATH).unlink()
        except OSError:
            pass


async def repair_one(*, source: str, name: str, signature: str, equation: str, equation_block: str, problem: str,
                     prompt: string.Template, chat: ChatFn, package: str = "") -> tuple[str | None, str]:
    """ONE repair of one function in ``source`` (a module's text), alone: ``(new module text, "")`` or ``(None, why)``.
    The request holds that function, its equation and the problem, not the script."""
    node = _find(source, name)
    current = ast.get_source_segment(source, node) if node is not None else signature
    reply = await chat(prompt.substitute(equation_block=equation_block, function_block=current or signature,
                                         problem=problem, related_block="(none)", package=package or "the package"))
    return replace_function(source, name, reply_code(reply), signature=signature, equation=equation)
