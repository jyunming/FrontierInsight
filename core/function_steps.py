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

The most the step can ask of the model in a quest is ``max_calls`` (``execution.code_function_steps_max_calls``): at
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
    return (f"{spec.signature.rstrip()}\n    \"\"\"{spec.purpose}\"\"\"\n    # {spec.equation}\n"
            f"    raise NotImplementedError(\"{_STUB_MARK}\")\n")


def initial_module(specs: list[Spec], summary: str = "") -> str:
    head = '"""' + (summary or "The model's equations, one function each; no scenario values.") + '"""\n\n'
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


def import_script(code_dir: Path, module: str, name: str) -> str:
    return _IMPORT_CHECK.format(code=str(Path(code_dir)), module=module, name=name)


# A third-party module that is not installed is a problem of the environment, not of the function.
_MISSING_MODULE = re.compile(r"(?:ModuleNotFoundError|ImportError): No module named '([\w.]+)'")


def environment_problem(stderr: str, package: str) -> str:
    """The name of a module the environment lacks (not the model's own package), or ``""``."""
    m = _MISSING_MODULE.search(stderr or "")
    if m and m.group(1).split(".")[0] != package:
        return m.group(1)
    return ""


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
    the step may make in the quest (``execution.code_function_steps_max_calls``) is spent."""
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

    async def _check(self, spec: Spec, record: dict[str, Any]) -> str:
        """``""`` when the function passes its checks, else what is wrong (never the expected value)."""
        text = self.model_path.read_text(encoding="utf-8")
        try:
            ast.parse(text)
        except SyntaxError as e:
            return f"model.py is not valid Python ({e.msg}, line {e.lineno})"
        module = f"{self.package}.model"
        rc, _out, err = await self.run(["-c", import_script(self.code, module, spec.name)])
        if rc != 0:
            missing = environment_problem(err, self.package)
            if missing:
                self.log.info("[implement] the import check of %s is skipped: this environment lacks %s", spec.name, missing)
            else:
                return "importing it failed:\n" + (err or "")[-1500:]
        rows = _eqt.example_rows(self.protocol)
        row = next((r for r in rows if r["id"] == spec.equation), None)
        if row is None or row["state"] != "ok":
            record.setdefault("untested", {})[spec.equation] = (row or {}).get("why") or "no worked example"
            return ""
        case_list = _eqt.cases([row], {spec.equation: {"module": module, "function": spec.name, "file": f"{self.package}/model.py"}})
        _eqt.write(self.code, case_list)
        rc, out, err = await self.run([_eqt.TEST_PATH, "--json", "--only", spec.equation])
        results = _eqt.parse_results(out)
        if results is None:
            return "the equation test did not run:\n" + (err or out or "")[-800:]
        bad = next((r for r in results if r.get("status") != _eqt.OK), None)
        return _eqt.failure_message(bad, row["formula"]) if bad else ""

    async def _fill(self, spec: Spec, record: dict[str, Any], specs: list[Spec]) -> str:
        """Fill ``spec`` and repair it alone until it passes or its repairs are spent. ``""`` when it passes, else why not."""
        entry = record["functions"][spec.name]
        source = self.model_path.read_text(encoding="utf-8")
        related = [f"{s.signature}   # {s.equation}: {s.purpose}" for s in specs
                   if s.name in spec.depends_on and s.name != spec.name]
        problem = ""
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
            self.model_path.write_text(new, encoding="utf-8")
            problem = await self._check(spec, record)
            if not problem:
                return ""
            source = new
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
            for s in specs:
                if record["functions"][s.name]["status"] == DONE and is_stub(on_disk, s.name):
                    record["functions"][s.name]["status"] = PENDING
                if _find(on_disk, s.name) is None:
                    self.model_path.write_text(on_disk.rstrip("\n") + "\n\n\n" + stub(s), encoding="utf-8")
                    on_disk = self.model_path.read_text(encoding="utf-8")
        save(self.root, record)
        for spec in specs:
            entry = record["functions"][spec.name]
            if entry["status"] == DONE:
                self.log.info("[implement] function %s (equation %s) is already written and checked", spec.name, spec.equation)
                continue
            self.log.info("[implement] writing function %s for equation %s", spec.name, spec.equation)
            problem = await self._fill(spec, record, specs)
            if problem:
                entry.update(status=FAILED, error=problem[:400])
                record.update(status="fell_back", reason=f"function {spec.name} (equation {spec.equation}) did not pass its checks")
                save(self.root, record)
                self.log.warning("[implement] function %s for equation %s still fails after %d repair(s) (%s); "
                                 "the code is written whole instead", spec.name, spec.equation, self.repairs,
                                 problem.splitlines()[0][:160])
                self._remove()
                return Outcome("fell_back", int(record.get("calls") or 0), record["reason"], spec.name)
            entry.update(status=DONE, error="")
            save(self.root, record)
            self.log.info("[implement] function %s passes its checks%s", spec.name,
                          "" if spec.equation not in (record.get("untested") or {}) else
                          f" (equation {spec.equation} has no worked example to test it with)")
        record["status"] = "done"
        save(self.root, record)
        return Outcome("done", int(record.get("calls") or 0))

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
