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
``repairs`` repairs per function. A function that still fails does NOT throw away the others: it is marked failed, the
step goes on with the remaining functions, and then the failed ones are left to up to :data:`MODULE_REPAIRS` focused
repairs of the whole module that name the functions that already pass (they must keep passing: every function is checked
again, and a repair that breaks one is undone). What is still failing then (the module imports, every function has
code) ends as ``partial``: the equation tests before the run, the whole-script repair and the honest stop of the
existing flow apply to it. Only when the module as a whole cannot be made to import, or a function has no code at all,
is the code written whole, as it always was (:func:`FunctionFiller.run_all` returns ``fell_back``).

A fault of the environment never costs a function a repair: a package the code imports is installed (FI's own
dependency path) and the check is run again; a check that could not run (a timeout, a harness that printed nothing, an
example whose input names are not the outline's parameters) is run once more and then left out, said in the record.

**Steps of the method.** The plan's non-equation steps (``protocol.model.steps``, ids ``S1``..., :mod:`core.plan_steps`) are
functions of the same package, one each: ``implements`` in the outline is the card's id, the function is written, labelled
(``# S1``) and import-checked like an equation's, tested by the card's worked example when it has a usable one (the same
judge as an equation's, so FI computes the expected value), and looked at statically (:mod:`core.step_static`: a function
that returns a number written into it, or never reads a parameter, fails). They live in ``model.py`` beside the equations,
not in a file of their own: the code-writing prompt, the focused module repair, the improve step and the layout checks all
read ``<package>/model.py``. A card with neither an example nor a relation is listed as ``untested_by_example`` in the
record. A function can be shown passages from the literature (``passages``, :mod:`core.step_passages`) and, for a step with an
example, the example's inputs; the expected output is shown only in the repair that follows a failed example.

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
from . import plan_steps as _plan_steps
from . import step_static as _step_static

RECORD = Path(".fi") / "function_steps.json"
#: Fixed limits (not settings): repairs of ONE function before the code is written whole; model functions above which the
#: code is written whole; the most requests the step (fills, repairs, and the repairs after the equation tests) may make
#: in a whole quest.
REPAIRS = 2
MODULE_REPAIRS = 2
MAX_FUNCTIONS = 16
MAX_CALLS = 40
#: Requests the step may make for ``n`` functions in a whole quest (one fill and up to :data:`REPAIRS` repairs, and a
#: little over for the module's repairs); the quest-wide count in the record (:func:`spend`) is what holds it. The caller
#: passes this as ``max_calls``; :data:`MAX_CALLS` stays the figure for the old equation-only call sites.
CALLS_PER_FUNCTION = 4
EQUATION, STEP = "equation", "step"
_STEP_ID = re.compile(r"S[0-9]+", re.IGNORECASE)


def max_calls_for(n_functions: int) -> int:
    """The most requests the step may make in a quest with ``n_functions`` functions to write."""
    return CALLS_PER_FUNCTION * max(1, int(n_functions))
#: A fenced Python block in a reply.
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
PENDING, DONE, FAILED = "pending", "done", "failed"
PARTIAL = "partial"
_STUB_MARK = "not written yet"


@dataclass
class Spec:
    name: str
    signature: str
    purpose: str
    equation: str  # the id of the equation, or of the step card (``S1``), the function computes
    depends_on: list[str] = field(default_factory=list)
    kind: str = EQUATION  # "equation" | "step"
    card: dict[str, Any] | None = None  # the step card (core/plan_steps.py) for a step

    def params(self) -> list[str]:
        node = ast.parse(self.signature.rstrip().rstrip(":") + ":\n    pass").body[0]
        return _param_names(node)  # type: ignore[arg-type]

    def required_params(self) -> list[str]:
        """The parameters a call must give (no default value, not ``*args`` or ``**kwargs``)."""
        a = ast.parse(self.signature.rstrip().rstrip(":") + ":\n    pass").body[0].args  # type: ignore[attr-defined]
        positional = [*a.posonlyargs, *a.args]
        required = positional[:len(positional) - len(a.defaults)]
        return [x.arg for x in required] + [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]


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
    signature that parses; otherwise no specs and the reason, and the code is written whole.

    Beside the equations' functions, an outline entry whose ``implements`` is the id of a step card
    (``protocol.model.steps``, ``S1``...; :mod:`core.plan_steps`) is a step's function. A step with no function, or whose
    function cannot be used, is a note and is left to the code written with the scenario: only the equations decide
    whether the step applies, as before. A step runs after the equations it ``uses``. A plan with no step cards gives
    exactly what it always gave."""
    equations = {_eqt._text(e["id"]).upper(): _eqt._text(e["id"]) for e in _eqt.eligible(protocol)}
    cards = {c["id"].upper(): c for c in _plan_steps.parse(protocol)[0] if _STEP_ID.fullmatch(c["id"])}
    rows = {r["id"]: r for r in _eqt.example_rows(protocol)}
    notes: list[str] = []
    specs: dict[str, Spec] = {}
    seen_eq: set[str] = set()
    seen_step: set[str] = set()
    for item in (outline or {}).get("functions") or []:
        if not isinstance(item, dict) or not _eqt._text(item.get("implements")):
            continue
        implements = _eqt._text(item["implements"])
        is_step = _STEP_ID.fullmatch(implements) is not None and implements.upper() not in equations
        eid = implements.upper() if is_step else equations.get(implements.upper())
        name = _eqt._text(item.get("name"))
        if is_step and implements.upper() not in cards:
            notes.append(f"the outline's function {name or '?'} implements {implements}, which the plan's steps do not list")
            continue
        if eid is None:
            notes.append(f"the outline's function {name or '?'} implements {item['implements']}, which the plan's model does not list")
            continue
        label = f"step {implements.upper()}" if is_step else f"equation {eid}"
        if (implements.upper() in seen_step) if is_step else (eid in seen_eq):
            notes.append(f"the outline names two functions for {label}")
            continue
        sig = _eqt._text(item.get("signature"))
        if not name.isidentifier() or not sig.startswith("def ") or not _signature_ok(sig, name):
            notes.append(f"the outline's signature for {label} cannot be read ({sig[:80] or 'none'})")
            continue
        if not is_step and eid in rows:  # the example's names, set by FI: the outline's own are only its reading of them
            sig = example_signature(sig, rows[eid])
        deps = [_eqt._text(d) for d in (item.get("depends_on") if isinstance(item.get("depends_on"), list) else [])]
        purpose = _eqt._text(item.get("one_line_purpose")) or name
        if is_step:
            card = cards[implements.upper()]
            wanted = _step_inputs_mismatch(card, sig)
            if wanted:
                notes.append(f"the outline's function {name} for {label} {wanted}")
                continue
            specs[name] = Spec(name=name, signature=sig, purpose=purpose, equation=card["id"],
                               depends_on=[d for d in deps if d], kind=STEP, card=card)
            seen_step.add(implements.upper())
        else:
            specs[name] = Spec(name=name, signature=sig, purpose=purpose, equation=eid, depends_on=[d for d in deps if d])
            seen_eq.add(eid)
    from . import oracle_check as _oracle

    missing = [i for i in _oracle.generating_equations(protocol) if _eqt._text(i) not in seen_eq]
    if missing:
        return [], [*notes, f"the outline names no function for equation {', '.join(missing)}"]
    if not specs:
        return [], notes or ["the outline names no function for an equation (`implements`)"]
    by_equation = {s.equation: s.name for s in specs.values() if s.kind == EQUATION}
    for spec in specs.values():  # a step is written after the equations it uses
        if spec.kind == STEP and spec.card:
            for u in spec.card.get("uses") or []:
                fn = by_equation.get(_eqt._text(u))
                if fn and fn not in spec.depends_on:
                    spec.depends_on.append(fn)
    for card_id, card in cards.items():
        if card_id not in seen_step:
            notes.append(f"step {card['id']} ({card['name'] or 'unnamed'}) has no function in the outline; it is written with the scenario")
    return _in_order(list(specs.values())), notes


def _step_inputs_mismatch(card: dict[str, Any], signature: str) -> str:
    """Why a step's function cannot be tested by the card's example (its parameters are not the card's inputs), or ``""``.
    Only a card with a usable example is held to it: that is the one case where the names are called by FI."""
    if card.get("example_state") != "ok":
        return ""
    try:
        node = ast.parse(signature.rstrip().rstrip(":") + ":\n    pass").body[0]
        params = {p for p in _param_names(node) if not p.startswith("*")}  # type: ignore[arg-type]
    except SyntaxError:
        return ""
    names = set(card.get("inputs") or {})
    extra = {"rng", "seed"} if card.get("random") not in (None, False, "", 0, "false", "False") else set()
    if names and params - extra != names:
        return (f"takes ({', '.join(sorted(params))}), not the card's inputs ({', '.join(sorted(names))}); "
                "its parameters are named as the card's inputs")
    return ""


def _signature_ok(sig: str, name: str) -> bool:
    try:
        node = ast.parse(sig.rstrip().rstrip(":") + ":\n    pass").body[0]
    except SyntaxError:
        return False
    return isinstance(node, ast.FunctionDef) and node.name == name


def example_signature(sig: str, row: dict[str, Any]) -> str:
    """``sig`` with the parameters of the plan's worked example for the equation it implements. The example's ``inputs`` are
    the plan's own record, frozen before any code is written; the outline is the code-writing model's reading of the same
    equation and can name the same quantities differently. The test calls the function by the example's names, so those
    are its parameters: FI sets them, rather than asking the plan to rename its example to the outline's words.

    A signature the test can already call (:func:`core.equation_tests.parameter_problems`: every input is a parameter, no
    parameter without a default is missing from the inputs) is returned as it was, as is one with ``*args``/``**kwargs`` or
    a row whose example is not usable. Otherwise the inputs come first, in the outline's order for the names it had
    (annotations kept) and the example's order for the rest, without defaults; then what the outline gave a default
    (a setting of the function, not an input) as it was; for a random equation ``rng`` when the outline gave none. An
    outline parameter with no default that is not an input of the example is dropped: the test cannot give it a value.
    A signature FI rewrote takes its parameters by keyword only (``def f(*, ...)``)."""
    if row.get("state") != "ok":
        return sig
    try:
        node = ast.parse(sig.rstrip().rstrip(":") + ":\n    pass").body[0]
    except SyntaxError:
        return sig
    if not isinstance(node, ast.FunctionDef):
        return sig
    a = node.args
    names = _param_names(node)
    if _eqt.parameter_problems([row], {row["id"]: (names, Spec(node.name, sig, "", "").required_params())}) == {}:
        return sig
    if a.vararg or a.kwarg:
        return sig
    positional = [*a.posonlyargs, *a.args]
    defaults: dict[str, ast.expr | None] = {x.arg: None for x in [*positional, *a.kwonlyargs]}
    for x, d in zip(positional[len(positional) - len(a.defaults):], a.defaults):
        defaults[x.arg] = d
    for x, d in zip(a.kwonlyargs, a.kw_defaults):
        defaults[x.arg] = d
    had = {x.arg: x for x in [*positional, *a.kwonlyargs]}
    inputs = list(row["inputs"])
    ordered = [n for n in had if n in inputs] + [n for n in inputs if n not in had]
    out = []
    for n in ordered:
        x = had.get(n)
        out.append(f"{n}: {ast.unparse(x.annotation)}" if x is not None and x.annotation is not None else n)
    fi_supplied = (*_eqt.RNG_PARAMS, _eqt.SEED_PARAM)  # given by the test for a random equation, whatever the outline wrote
    for n, x in had.items():
        if n not in inputs and (defaults[n] is not None or (row.get("random") and n in fi_supplied)):
            d = defaults[n]
            out.append(ast.unparse(x) if d is None else
                       f"{ast.unparse(x)}{' = ' if x.annotation is not None else '='}{ast.unparse(d)}")
    if row.get("random") and not any(n in fi_supplied for n in had):
        out.append("rng")
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    # Keyword-only: code written against the outline's order (a resumed quest, a function repaired alone) then fails at
    # the call, loudly, instead of passing its values to the wrong parameters.
    return f"def {node.name}(*, {', '.join(out)}){returns}:"


def align_outline(outline: dict[str, Any] | None, protocol: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """``(outline, what changed)``: the outline with each equation's function given the parameters of the plan's example
    (:func:`example_signature`), in ``functions[].signature`` and in the scaffold's ``def`` line, so the request that
    writes the study shows the signature the tests and the repairs hold the function to. The outline given is not changed;
    the sentences name the function, the equation and the old and new parameters."""
    if not isinstance(outline, dict) or not isinstance(outline.get("functions"), list):
        return outline or {}, []
    rows = {r["id"].upper(): r for r in _eqt.example_rows(protocol)}
    equations = {_eqt._text(e["id"]).upper() for e in _eqt.eligible(protocol)}
    functions: list[Any] = []
    notes: list[str] = []
    scaffold = outline.get("scaffold")
    for item in outline["functions"]:
        row = rows.get(_eqt._text(item.get("implements")).upper()) if isinstance(item, dict) else None
        sig = _eqt._text(item.get("signature")) if row is not None else ""
        name = _eqt._text(item.get("name")) if row is not None else ""
        if (row is None or _eqt._text(item.get("implements")).upper() not in equations or not sig.startswith("def ")
                or not _signature_ok(sig, name)):
            functions.append(item)
            continue
        new = example_signature(sig, row)
        if new == sig:
            functions.append(item)
            continue
        functions.append({**item, "signature": new})
        notes.append(f"{name} ({row['id']}): ({sig[sig.index('(') + 1:sig.rindex(')')]}) -> "
                     f"({new[new.index('(') + 1:new.rindex(')')]})")
        if isinstance(scaffold, str):
            scaffold = _scaffold_with_signature(scaffold, name, new)
    if not notes:
        return outline, []
    return {**outline, "functions": functions, **({"scaffold": scaffold} if isinstance(scaffold, str) else {})}, notes


def _scaffold_with_signature(scaffold: str, name: str, signature: str) -> str:
    """The scaffold with the ``def`` header of ``name`` replaced by ``signature`` (kept as it was when the scaffold does not
    parse or the header and the first statement share a line)."""
    try:
        tree = ast.parse(scaffold)
    except SyntaxError:
        return scaffold
    node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None or not node.body or node.body[0].lineno <= node.lineno:
        return scaffold
    lines = scaffold.splitlines(keepends=True)
    start = node.decorator_list[-1].end_lineno if node.decorator_list else node.lineno - 1
    indent = " " * node.col_offset
    lines[start:node.body[0].lineno - 1] = [indent + signature + "\n"]
    return "".join(lines)


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
    example) and step cards they implement. A different one means the functions are written again. A set of specs
    with no step hashes as it did before steps existed."""
    wanted = {s.equation for s in specs if s.kind == EQUATION}
    eqs = [{k: e.get(k) for k in ("id", "formula", "example")} for e in _eqt.eligible(protocol)
           if _eqt._text(e["id"]) in wanted]
    body: dict[str, Any] = {"f": [[s.name, s.signature, s.equation, s.depends_on] for s in specs], "e": eqs}
    cards = [{k: s.card.get(k) for k in ("id", "name", "does", "inputs", "output", "uses", "source", "example", "relations")}
             for s in specs if s.kind == STEP and s.card]
    if cards:
        body["s"] = cards
    blob = json.dumps(body, sort_keys=True, default=str)
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


def _with_stub(source: str, spec: Spec) -> str | None:
    """``source`` with the function of ``spec`` (read from the syntax tree, or from the lines when the module does not
    parse) replaced by its placeholder; ``None`` when the module has no such function."""
    node = _find(source, spec.name)
    lines = source.splitlines(keepends=True)
    if node is not None:
        first, last = _span(node)
    else:
        found = next(((a, b) for a, b, name in _text_spans(source) if name == spec.name), None)
        if found is None:
            return None
        first, last = found
    return "".join(lines[: first - 1]) + stub(spec) + chr(10) * 2 + "".join(lines[last:])


def is_stub(source: str, name: str) -> bool:
    node = _find(source, name)
    if node is None:
        return True
    return any(isinstance(n, ast.Raise) and _STUB_MARK in ast.dump(n) for n in ast.walk(node))


def _name_key(name: str) -> str:
    return name.lower().replace("_", "")


def pair_parameter_names(have: list[str], want: list[str]) -> dict[str, str] | None:
    """``{name the function has: name it must have}`` when the two lists of unmatched names pair up with no choice to
    make, else ``None``. ``have`` are the function's parameters that are not wanted, ``want`` the wanted names the function
    lacks. A pair is made when the two names are the same word apart from capitals and underscores (``rate`` and ``Rate``), and
    when exactly one name is left on each side. Two or more names left on a side, or lists of different lengths, are never
    paired: that would be a guess. The caller's test of the function's value is what judges the renamed function."""
    if not have or len(have) != len(want):
        return None
    left_h, left_w, pairs = list(have), list(want), {}
    for h in list(left_h):
        same = [w for w in left_w if _name_key(w) == _name_key(h)]
        rivals = [x for x in left_h if _name_key(x) == _name_key(h)]
        if len(same) == 1 and len(rivals) == 1:
            pairs[h] = same[0]
            left_h.remove(h)
            left_w.remove(same[0])
    if len(left_h) == 1 and len(left_w) == 1:
        pairs[left_h[0]] = left_w[0]
    elif left_h or left_w:
        return None
    return pairs


def rename_parameters(source: str, name: str, renames: dict[str, str], *, callers: list[str] | None = None) -> str | None:
    """``source`` with the parameters of the top-level function ``name`` renamed (``renames``: old to new) in its ``def``
    line and wherever its body reads them, or ``None`` when that cannot be done without changing what the code means: the
    function is not there, an old name is not one of its parameters, a new name is already used inside the function, an
    inner function or lambda has a parameter of the same name, or some code (``callers``: more module texts, besides
    ``source``) calls the function with a keyword of an old name. Only the text of the names changes; the result is checked
    to be the same code under the new names."""
    node = _find(source, name)
    if node is None or not renames:
        return None
    a = node.args
    if a.vararg or a.kwarg:
        return None
    own = {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]}
    if not set(renames) <= own or len(set(renames.values())) != len(renames):
        return None
    used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.arg for n in ast.walk(node) if isinstance(n, ast.arg)}
    if (used - set(renames)) & set(renames.values()):
        return None
    for n in ast.walk(node):
        if isinstance(n, (ast.Global, ast.Nonlocal)) and set(n.names) & set(renames):
            return None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not node and (
                {x.arg for x in ast.walk(n.args) if isinstance(x, ast.arg)} & set(renames)):
            return None
    for text in [source, *(callers or [])]:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        for c in ast.walk(tree):
            if isinstance(c, ast.Call) and ((isinstance(c.func, ast.Name) and c.func.id == name)
                                            or (isinstance(c.func, ast.Attribute) and c.func.attr == name)):
                if any(k.arg in renames for k in c.keywords):
                    return None
    spots: list[tuple[int, int, str, str]] = []  # (line, utf-8 column, old, new)
    for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
        if x.arg in renames:
            spots.append((x.lineno, x.col_offset, x.arg, renames[x.arg]))
    for stmt in node.body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name) and n.id in renames:
                spots.append((n.lineno, n.col_offset, n.id, renames[n.id]))
    lines = source.splitlines(keepends=True)
    for line, col, old, new in sorted(spots, reverse=True):
        raw = lines[line - 1].encode("utf-8")
        if raw[col:col + len(old)] != old.encode("utf-8"):
            return None
        lines[line - 1] = (raw[:col] + new.encode("utf-8") + raw[col + len(old):]).decode("utf-8")
    out = "".join(lines)

    expected = ast.parse(source)
    target = next(n for n in expected.body if isinstance(n, ast.FunctionDef) and n.name == name)
    for x in [*target.args.posonlyargs, *target.args.args, *target.args.kwonlyargs]:
        x.arg = renames.get(x.arg, x.arg)
    for stmt in target.body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name) and n.id in renames:
                n.id = renames[n.id]
    try:
        return out if ast.dump(ast.parse(out)) == ast.dump(expected) else None
    except SyntaxError:
        return None


def match_parameter_names(source: str, name: str, wanted: list[str], *, callers: list[str] | None = None) -> str | None:
    """``source`` with the parameters of function ``name`` renamed to ``wanted`` (the parameters the function must have), when
    the only difference is names that pair with no choice (:func:`pair_parameter_names`); ``None`` when there is nothing to
    rename or no safe rename."""
    node = _find(source, name)
    if node is None:
        return None
    have = [n for n in _param_names(node) if not n.startswith("*")]
    extra = [n for n in have if n not in wanted]
    lacking = [n for n in wanted if n not in have and not n.startswith("*")]
    pairs = pair_parameter_names(extra, lacking)
    return rename_parameters(source, name, pairs, callers=callers) if pairs else None


def match_example_names(source: str, name: str, inputs: list[str], *, random: bool = False,
                        callers: list[str] | None = None) -> tuple[str | None, dict[str, str]]:
    """``(source with the parameters of function ``name`` named as the plan's worked example names its inputs, the renames
    made)``, or ``(None, {})`` when there is nothing to do or no safe rename. The test calls the function by the example's
    input names (:func:`core.equation_tests.parameter_problems`): an input the function has no parameter for is paired with a
    parameter that has no default and is no input (a random equation's ``rng``/``seed`` are FI's), when that pairing leaves
    no choice (:func:`pair_parameter_names`). A parameter with a default is a setting and is never renamed."""
    node = _find(source, name)
    if node is None or node.args.vararg or node.args.kwarg:
        return None, {}
    a = node.args
    positional = [*a.posonlyargs, *a.args]
    required = [x.arg for x in positional[:len(positional) - len(a.defaults)]]
    required += [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
    everything = {x.arg for x in [*positional, *a.kwonlyargs]}
    given = set(inputs) | ({*_eqt.RNG_PARAMS, _eqt.SEED_PARAM} if random else set())
    lacking = [n for n in inputs if n not in everything]
    extra = [n for n in required if n not in given]
    pairs = pair_parameter_names(extra, lacking) if lacking else None
    renamed = rename_parameters(source, name, pairs, callers=callers) if pairs else None
    return (renamed, pairs or {}) if renamed is not None else (None, {})


def changed_signatures(source: str, fixed: dict[str, str]) -> list[str]:
    """For each function of ``fixed`` (its name to the ``def`` line the outline gave it) that ``source`` defines at the
    top level with other parameters: a plain sentence saying so. The outline is the one place a function's parameters
    are fixed: the tests FI writes before the run call the function by them, so a repair that changes them makes the
    next test fail on names alone (and the next repair change them back). Only names are compared."""
    out = []
    for name, signature in fixed.items():
        node = _find(source, name)
        if node is None:
            continue
        want = _param_names(ast.parse(signature.rstrip().rstrip(":") + ":\n    pass").body[0])  # type: ignore[arg-type]
        have = _param_names(node)
        if have != want:
            out.append(f"`{name}` must keep the parameters ({', '.join(want)}) the outline fixed; the repair gave "
                       f"({', '.join(have)})")
    return out


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
        # A reply that only names a parameter otherwise (``width_max`` for ``width``) is the function with its names put
        # right by FI, when there is no choice in the pairing; any other difference is refused.
        renamed = match_parameter_names(reply, name, want, callers=[source])
        if renamed is not None:
            reply = renamed
            new_tree = ast.parse(reply)
            node = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
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
    if record.get("package") != package or record.get("status") not in ("done", PARTIAL):
        return False
    return (Path(quest_root) / "code" / package / "model.py").is_file()


def counts(record: dict[str, Any]) -> tuple[int, int]:
    """``(functions that finished done, functions in all)`` from a record of the step."""
    entries = [e for e in (record.get("functions") or {}).values() if isinstance(e, dict)]
    return sum(1 for e in entries if e.get("status") == DONE), len(entries)


def failing(record: dict[str, Any]) -> list[str]:
    """The names of the functions a record of the step leaves failing."""
    return [n for n, e in (record.get("functions") or {}).items() if isinstance(e, dict) and e.get("status") == FAILED]


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
#: Installs packages (pip names) into the quest's environment; returns the ones that could not be, each with the reason
#: (the engine's own :meth:`_install_packages`).
InstallFn = Callable[[list[str]], Awaitable[list[tuple[str, str]]]]


def harness_trouble(rc: int, err: str) -> bool:
    """Whether a command did not run to a verdict because of the machinery, not the code: it could not start, or it timed
    out."""
    return rc == -1 or "(timed out)" in (err or "")


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
                 max_calls: int = 30, constants: str = "", install: InstallFn | None = None,
                 module_repairs: int = MODULE_REPAIRS, declared: list[str] | None = None,
                 topic_block: str = "", passages: dict[str, str] | None = None) -> None:
        self.topic_block = topic_block or "(the topic is not available)"
        #: ``{equation or step id: passages from the literature}`` (core/step_passages.py), found by the caller before the
        #: step starts; an id with none gets an empty literature block.
        self.passages = {str(k): str(v) for k, v in (passages or {}).items() if str(v).strip()}
        self._by_id: dict[str, Spec] = {}
        self._example_failed: set[str] = set()
        self.root, self.package, self.protocol, self.summary = Path(quest_root), package, protocol, summary
        self.prompts, self.chat, self.run, self.log = prompts, chat, run, log
        self.install, self.module_repairs = install, max(0, int(module_repairs))
        self._tried_installs: set[str] = set()
        self.declared = [str(d) for d in declared or []]
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
        spec = self._by_id.get(eid)
        if spec is not None and spec.kind == STEP and spec.card:
            return self._step_block(spec.card)
        e = self._equation(eid)
        lines = [f"{eid}: {_eqt._text(e.get('formula'))}"]
        if _eqt._text(e.get("source")):
            lines.append(f"source: {_eqt._text(e.get('source'))}")
        if _eqt._text(e.get("derivation")):
            lines.append(f"derivation: {_eqt._text(e.get('derivation'))}")
        return "\n".join(lines)

    def _step_block(self, card: dict[str, Any]) -> str:
        """What a step of the method is, for the request that writes its function (never an expected value)."""
        lines = [f"{card['id']} (a step of the method, not an equation): {card.get('name') or ''}".rstrip(),
                 f"what it does: {card.get('does') or '(not stated)'}"]
        inputs = card.get("inputs") or {}
        if inputs:
            lines.append("inputs: " + "; ".join(f"{k} ({v})" if v else k for k, v in inputs.items()))
        if card.get("output"):
            lines.append(f"returns: {card['output']}")
        for u in card.get("uses") or []:
            e = self._equation(_eqt._text(u))
            if _eqt._text(e.get("formula")):
                lines.append(f"uses equation {u}: {_eqt._text(e.get('formula'))}  (call its function; do not write it again)")
        if card.get("source"):
            lines.append(f"source: {card['source']}")
        if card.get("relations"):
            lines.append("properties its result must have: " + "; ".join(card["relations"]))
        return "\n".join(lines)

    def _step_row(self, spec: Spec) -> dict[str, Any] | None:
        """The worked example of a step card as :func:`core.equation_tests.example_rows` judges an equation's (FI computes
        the expected value from the card's formula), or ``None`` when the card has none."""
        card = spec.card or {}
        if card.get("example") is None:
            return None
        pseudo = {"model": {"equations": [{"id": spec.equation, "formula": card.get("does") or card.get("name") or "step",
                                           "role": "analyses", "example": card["example"], "random": card.get("random")}]}}
        return next((r for r in _eqt.example_rows(pseudo) if r["id"] == spec.equation), None)

    def _literature_block(self, spec: Spec) -> str:
        text = self.passages.get(spec.equation, "").strip()
        if not text:
            return ""
        return ("# What the literature says (passages from the source the plan cites; use them for the method's definition "
                "and details, do not copy numbers from them)\n\n" + text + "\n")

    def _example_block(self, spec: Spec, *, failed: bool = False) -> str:
        """A step's worked inputs; for the repair that follows a failed example, that its result did not match. Never the
        output FI computed for them or the tolerance: FI judges this very case, so a function shown its answer could
        special-case it."""
        if spec.kind != STEP:
            return ""
        row = self._step_row(spec)
        if row is None or row["state"] != "ok":
            return ""
        given = ", ".join(f"{k} = {v:g}" for k, v in row["inputs"].items())
        text = f"# A worked case\n\nFI will call the function with: {given}.\n"
        if row.get("random"):
            text += "Its result is a random draw: FI compares the mean of many draws.\n"
        elif failed:
            text += ("FI worked out the result for these inputs from the plan's own formula, and what the function as it "
                     "is returns does not match it. Find out why from the step's description.\n")
        return text

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
        argv = ["-c", import_script("code", module, spec.name)]
        retried = False
        while True:
            rc, _out, err = await self.run(argv)
            if rc == 0:
                break
            if harness_trouble(rc, err):
                # The check could not run (a timeout, a command that did not start): once more, and then left out. It says
                # nothing about the function, and costs it nothing.
                if not retried:
                    retried = True
                    continue
                self._unchecked(record, spec, f"the import check could not be run here ({exception_line(err, 120)})")
                return "", "", ""
            missing = environment_problem(err, self.package)
            if missing:
                top = missing.split(".")[0]
                if self.install is not None and top not in self._tried_installs:
                    self._tried_installs.add(top)  # a package is asked for once in a run, found or not
                    if await self._install_module(top, record):
                        continue
                # A package the environment lacks and cannot get: the function asked for something that is not there.
                return (f"it imports `{missing}`, which is not installed in this environment and could not be installed; "
                        "use only the standard library and the packages the plan names"), spec.name, traceback_end(err, 600)
            who, line = locate_failure(err, text, self.package, spec.name, known)
            end = traceback_end(err, 1500)
            if who == spec.name:
                return "importing it failed" + (f" (line {line})" if line else "") + ":\n" + end, who, end
            return (f"importing the module failed {place(who, line)}, not in `{spec.name}`:\n" + end), who, end
        if spec.kind == STEP:
            row = self._step_row(spec)
            if row is None or row["state"] != "ok":
                record.setdefault("untested", {})[spec.equation] = (row or {}).get("why") or "no worked example"
            else:
                problem = await self._example_test(spec, row, record, module, formula=str((spec.card or {}).get("does") or ""))
                if problem:
                    self._example_failed.add(spec.equation)
                    return problem, spec.name, ""
                self._example_failed.discard(spec.equation)
            why = _step_static.check(text, spec.name)
            return (why, spec.name, "") if why else ("", "", "")
        rows = _eqt.example_rows(self.protocol)
        row = next((r for r in rows if r["id"] == spec.equation), None)
        if row is None or row["state"] != "ok":
            record.setdefault("untested", {})[spec.equation] = (row or {}).get("why") or "no worked example"
            return "", "", ""
        problem = await self._example_test(spec, row, record, module, formula=row["formula"])
        return (problem, spec.name, "") if problem else ("", "", "")

    async def _example_test(self, spec: Spec, row: dict[str, Any], record: dict[str, Any], module: str, *,
                            formula: str) -> str:
        """Run the worked example of ``row`` against the function; ``""`` when it passes or could not be judged (said in the
        record), else the plain problem (never the expected value)."""
        rel = f"{self.package}/model.py"
        try:
            text = (self.code / rel).read_text(encoding="utf-8")
        except OSError:
            text = ""
        # FI's own reading of the function (core/equation_tests.static_facts) decides whether it can be called with the
        # example's inputs and whether it returns its outputs by name; nothing the run prints can claim either.
        case_list = _eqt.cases([row], {spec.equation: {"module": module, "function": spec.name, "file": rel,
                                                       "static": _eqt.static_facts(text, spec.name)}})
        _eqt.write(self.code, case_list)
        results: list[dict[str, Any]] = []
        for _try in range(2):  # a run that printed no result is run once more; a second time it fails the test
            # The function is called in a process given only its inputs; FI judges what it returns here, with the
            # example's expected value that process never had (core/equation_measure.py).
            nonce = _eqt.new_nonce()
            _rc, out, _err = await self.run(_eqt.measure_argv(case_list, nonce=nonce, only=[spec.equation]))
            results = _eqt.judged(case_list, out, nonce=nonce, only=[spec.equation])
            if not any(r.get("detail") == _eqt.NO_RESULT for r in results):
                break
        bad = next((r for r in results if r.get("status") != _eqt.OK), None)
        if bad and bad.get("status") in (_eqt.CONTRACT, _eqt.OUTPUTS):
            # FI's own reading: the plan's example names inputs the outline's function does not take, and its parameters
            # are fixed; or it gives one value where the function returns its outputs by name: nothing the function could
            # change. FI's own mismatch with the plan, not the function's; the equation tests before the run ask the plan.
            self._unchecked(record, spec, f"the plan's worked example does not fit the outline's function ({bad.get('detail')})")
            return ""
        if not bad:
            return ""
        message = _eqt.failure_message(bad, formula)
        if spec.kind == STEP:
            message = message.replace(f"implements equation {spec.equation} of the model", f"implements step {spec.equation} of the method")
        return message

    def _unchecked(self, record: dict[str, Any], spec: Spec, why: str) -> None:
        """A check that could not be made (a fault of the environment or of FI's own test): said in the record and the log,
        and it costs the function nothing."""
        record.setdefault("untested", {})[spec.equation] = why
        record.setdefault("unchecked", {})[spec.name] = why
        self.log.info("[implement] function %s is not checked: %s", spec.name, why)

    async def _install_module(self, module: str, record: dict[str, Any]) -> bool:
        """Install the package a module name comes from, through the engine's own dependency path; ``True`` when it is
        there now."""
        from . import experiment_deps as _deps

        # The policy of FI's own dependency install (core/experiment_deps.py): a package whose import name FI knows
        # (IMPORT_TO_PIP), or one the plan or the design declares. A name the generated code merely imports is never
        # installed: the function asked for something that is not available.
        pip = _deps.pip_name(module)
        if pip is None:
            for dep in self.declared:
                if _deps.normalize(_deps.requirement_name(dep) or dep) == _deps.normalize(module):
                    pip = dep
                    break
        if pip is None:
            self.log.info("[implement] %s is not installed, and neither FI's known packages nor the plan's dependencies "
                          "name it; not installing it", module)
            return False
        try:
            failed = await self.install([pip]) if self.install is not None else [(pip, "no installer")]
        except Exception as e:  # noqa: BLE001 -- an install that could not be asked for leaves the package missing
            self.log.warning("[implement] could not install %s for the function checks: %r", pip, e)
            return False
        record.setdefault("installed", [])
        if not failed:
            record["installed"].append(pip)
            self.log.info("[implement] installed %s for the function checks", pip)
        return not failed

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
                    prompt=self.prompts["repair"], chat=self.chat, package=self.package,
                    topic_block=self.topic_block, literature_block=self._literature_block(other))
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
        text = _with_stub(source, other)
        if text is None:
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
        problem = who = trace = ""
        own: set[str] = set()
        for attempt in range(1 + self.repairs):
            if not self._spend(record):
                return "the most requests the step may make in this quest are spent"
            entry["attempts"] = int(entry.get("attempts") or 0) + 1
            if attempt == 0:
                prompt = self.prompts["fill"].substitute(
                    model_block=self.summary or "(not stated)", equation_block=self._equation_block(spec.equation),
                    function_block=f"{spec.signature}\n    # {spec.purpose}", related_block=_text_block(related),
                    constants_block=self.constants or "(none)", package=self.package,
                    topic_block=self.topic_block, literature_block=self._literature_block(spec),
                    example_block=self._example_block(spec))
            else:
                current = _find(source, spec.name)
                prompt = self.prompts["repair"].substitute(
                    equation_block=self._equation_block(spec.equation),
                    function_block=ast.get_source_segment(source, current) if current is not None else spec.signature,
                    problem=problem, related_block=_text_block(related), package=self.package,
                    topic_block=self.topic_block, literature_block=self._literature_block(spec),
                    example_block=self._example_block(spec, failed=spec.equation in self._example_failed))
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
        if problem and trace and who == spec.name:
            # Its code does not even import: the module goes back to the function's placeholder, so that the other
            # functions can be written and checked, and the focused repair of the module has the problem to read.
            stubbed = _with_stub(self.model_path.read_text(encoding="utf-8"), spec)
            if stubbed is not None:
                self.model_path.write_text(stubbed, encoding="utf-8")
        return problem

    async def run_all(self, specs: list[Spec]) -> Outcome:
        """Write every function of ``specs`` in order. Returns ``done`` when all pass, ``fell_back`` when one ends the
        step (its files are removed, the code is then written whole), ``skipped`` when it was not used."""
        digest = spec_hash(specs, self.protocol)
        self._by_id = {s.equation: s for s in specs}
        record = load(self.root)
        if record.get("package") != self.package or record.get("hash") != digest:
            record = {"calls_total": int(record.get("calls_total") or 0), "package": self.package, "hash": digest,
                      "functions": {}, "calls": 0, "status": "running"}
        elif record.get("status") == "fell_back":
            return Outcome("skipped", reason=str(record.get("reason") or "an earlier pass of the step fell back"))
        record["functions"] = {s.name: record["functions"].get(s.name) or
                               {"equation": s.equation, "status": PENDING, "attempts": 0} for s in specs}
        for s in specs:
            record["functions"][s.name]["kind"] = s.kind
        # A step with neither a usable worked example nor a relation to hold is not tested by anything but the static look;
        # said here so that the checks that come after can pick it up.
        record["untested_by_example"] = [s.equation for s in specs if s.kind == STEP and s.card
                                         and s.card.get("untested_by_example")]
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
                if entry["status"] == FAILED:
                    continue  # its repairs are spent: it is left to the focused repair of the module, below
                self.log.info("[implement] writing function %s for equation %s", spec.name, spec.equation)
                problem = await self._fill(spec, record, specs)
                if problem:
                    if record.get("blamed"):
                        return self._give_up(record, spec, problem)  # the module itself cannot be made to import
                    entry.update(status=FAILED, error=traceback_end(problem, 800))
                    save(self.root, record)
                    self.log.warning("[implement] function %s for equation %s still fails after %d repair(s) (%s ... %s); "
                                     "the other functions are written and it is left to a repair of the module",
                                     spec.name, spec.equation, self.repairs, (problem.splitlines() or [""])[0][:120],
                                     exception_line(problem, 200))
                    continue
                entry.update(status=DONE, error="")
                save(self.root, record)
                self.log.info("[implement] function %s passes its checks%s", spec.name,
                              "" if spec.equation not in (record.get("untested") or {}) else
                              f" (equation {spec.equation} has no worked example to test it with)")
            if all(record["functions"][s.name]["status"] in (DONE, FAILED) for s in specs):
                break
        else:
            left = next(s for s in specs if record["functions"][s.name]["status"] not in (DONE, FAILED))
            return self._give_up(record, left, "it could not be written again after the module's import failed inside another function")
        if failing(record):
            await self._repair_module(specs, record)
        return self._finish(specs, record)

    def _finish(self, specs: list[Spec], record: dict[str, Any]) -> Outcome:
        """The step's end: ``done`` when every function passes, ``partial`` when the module imports and every function
        has code but some still fail their test, else the code is written whole."""
        source = self.model_path.read_text(encoding="utf-8")
        no_code = [s for s in specs if record["functions"][s.name]["status"] != DONE and is_stub(source, s.name)]
        done, total = counts(record)
        record.update(functions_done=done, functions_total=total)
        if no_code:
            first = no_code[0]
            return self._give_up(record, first, str(record["functions"][first.name].get("error") or "it has no code")
                                 + "\n(the module could not be made to import with it)")
        record["status"] = "done" if done == total else PARTIAL
        if record["status"] == PARTIAL:
            self.log.warning("[implement] the model's package is written, but %d of %d function(s) still fail their check "
                             "(%s); the tests before the run decide what happens next", total - done, total,
                             ", ".join(failing(record)))
        save(self.root, record)
        return Outcome(record["status"], int(record.get("calls") or 0), failed=", ".join(failing(record)))

    # -- the focused repair of the module --

    def _module_blocks(self, specs: list[Spec], record: dict[str, Any]) -> tuple[str, str]:
        passing = [f"- `{s.name}`  ({s.equation}): {s.signature}" for s in specs
                   if record["functions"][s.name]["status"] == DONE]
        broken = []
        for s in specs:
            entry = record["functions"][s.name]
            if entry["status"] != FAILED:
                continue
            broken.append(f"## `{s.name}`  (equation {s.equation})\n{self._equation_block(s.equation)}\n"
                          f"{s.signature}\nWhat is wrong with it: {entry.get('error') or 'it fails its check'}")
        return _text_block(passing), "\n\n".join(broken)

    def _with_labels(self, text: str, specs: list[Spec]) -> str:
        """``text`` with the equation's id in a comment above each function of ``specs`` that lost it."""
        from . import oracle_check as _oracle

        for s in specs:
            node = _find(text, s.name)
            if node is None:
                continue
            first = _span(node)[0]
            lines = text.splitlines(keepends=True)
            near = "".join(lines[max(0, first - 3): node.end_lineno or first])
            if not _oracle._labelled_in(s.equation, near):
                lines.insert(first - 1, f"# {s.equation}\n")
                text = "".join(lines)
        return text

    def _module_ok(self, text: str, specs: list[Spec]) -> str:
        """Why a reply's module cannot be used (``""`` when it can): it parses, and every function has the parameters the
        outline fixed."""
        if not text.strip():
            return "the reply held no code"
        try:
            ast.parse(text)
        except SyntaxError as e:
            return f"the reply is not valid Python ({e.msg}, line {e.lineno})"
        for s in specs:
            node = _find(text, s.name)
            if node is None:
                return f"the reply holds no function named `{s.name}`"
            if _param_names(node) != s.params():
                return f"the parameters of `{s.name}` must be exactly ({', '.join(s.params())})"
        return ""

    async def _repair_module(self, specs: list[Spec], record: dict[str, Any]) -> None:
        """The functions that failed their repairs are left to the repair of the whole module (at most ``module_repairs``
        requests): the model is shown the module, the failed functions with what is wrong, and the names of the functions
        that already pass, which must keep passing. Every function is checked again afterwards; a repair that breaks a
        function that passed is undone."""
        if "module" not in self.prompts:
            return
        while failing(record) and int(record.get("module_repairs") or 0) < self.module_repairs:
            if not self._spend(record):
                self.log.warning("[implement] the most requests the step may make in this quest are spent; the module is not "
                                 "repaired")
                return
            record["module_repairs"] = int(record.get("module_repairs") or 0) + 1
            save(self.root, record)
            source = self.model_path.read_text(encoding="utf-8")
            passing_block, failing_block = self._module_blocks(specs, record)
            prompt = self.prompts["module"].substitute(package=self.package, passing_block=passing_block,
                                                        failing_block=failing_block, module=source.rstrip("\n"),
                                                        topic_block=self.topic_block)
            self.log.info("[implement] repairing the module for %s; %s must keep passing", ", ".join(failing(record)),
                          ", ".join(s.name for s in specs if record["functions"][s.name]["status"] == DONE) or "no function")
            try:
                reply = await self.chat(prompt)
            except Exception as e:  # noqa: BLE001 -- a request that failed is a repair that did not happen
                self.log.warning("[implement] the repair of the module could not be asked for: %r", e)
                continue
            new = self._with_labels(reply_code(reply), specs)
            why = self._module_ok(new, specs)
            if why:
                self.log.warning("[implement] the repair of the module was not used: %s", why)
                continue
            self.model_path.write_text(new.rstrip("\n") + "\n", encoding="utf-8")
            known = _dumps(new)
            verdicts: dict[str, str] = {}
            for s in specs:
                problem, _who, _trace = await self._check(s, record, known)
                verdicts[s.name] = problem
            broken = [s.name for s in specs if record["functions"][s.name]["status"] == DONE and verdicts[s.name]]
            if broken:
                self.model_path.write_text(source, encoding="utf-8")
                self.log.warning("[implement] the repair of the module was undone: %s passed before and does not now",
                                 ", ".join(broken))
                continue
            for s in specs:
                entry = record["functions"][s.name]
                if entry["status"] == FAILED:
                    if is_stub(new, s.name):
                        entry["error"] = entry.get("error") or "it has no code"  # a placeholder imports; it is not written
                    elif verdicts[s.name]:
                        entry["error"] = traceback_end(verdicts[s.name], 800)
                    else:
                        entry.update(status=DONE, error="")
                        self.log.info("[implement] function %s passes its checks after the repair of the module", s.name)
            save(self.root, record)

    def _give_up(self, record: dict[str, Any], spec: Spec, problem: str) -> Outcome:
        """The step ends: the error is recorded where it is (the END of the traceback, and whose it is), the half-written
        package is removed and the code is written whole."""
        blamed = str(record.pop("blamed", "") or "")
        target = spec
        if blamed in record["functions"] and blamed != spec.name:
            target = Spec(name=blamed, signature="", purpose="", equation=str(record["functions"][blamed].get("equation")))
        entry = record["functions"][target.name]
        entry.update(status=FAILED, error=traceback_end(problem, 800))
        done, total = counts(record)
        record.update(functions_done=done, functions_total=total)
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

    # -- after the step: the second-version check asks for one function to be tested again, or repaired once --

    def know(self, specs: list[Spec]) -> None:
        """Tell a filler that did not write the functions (a later pass of the quest) which functions the package has."""
        self._by_id = {s.equation: s for s in specs}

    async def still_passes(self, spec: Spec) -> bool:
        """The function's own checks (import, worked example, the static look) again on what is in the module now. A
        check that could not be made does not count as a pass."""
        record: dict[str, Any] = {}
        problem, _who, _trace = await self._check(spec, record)
        return not problem and spec.name not in (record.get("unchecked") or {})

    async def repair_function(self, spec: Spec, problem: str) -> bool:
        """ONE repair of one function, alone (its card or equation and the problem, never the expected value), written into
        the module when the reply is a usable function with the parameters it had. ``True`` when the module changed."""
        source = self.model_path.read_text(encoding="utf-8")
        record = load(self.root)
        if not self._spend(record):  # counted in the quest's total before the request, like every other one of the step
            self.log.info("[implement] function %s is not repaired: the most requests the step may make in this quest are spent",
                          spec.name)
            return False
        new, why = await repair_one(
            source=source, name=spec.name, signature=spec.signature, equation=spec.equation,
            equation_block=self._equation_block(spec.equation), problem=problem, prompt=self.prompts["repair"],
            chat=self.chat, package=self.package, topic_block=self.topic_block,
            literature_block=self._literature_block(spec), example_block=self._example_block(spec))
        if new is None:
            self.log.info("[implement] the repair of function %s was not used: %s", spec.name, why)
            return False
        if new == source:
            return False
        self.model_path.write_text(new, encoding="utf-8")
        return True

    def _remove(self) -> None:
        import shutil

        shutil.rmtree(self.code / self.package, ignore_errors=True)
        try:
            (self.code / _eqt.TEST_PATH).unlink()
        except OSError:
            pass


async def repair_one(*, source: str, name: str, signature: str, equation: str, equation_block: str, problem: str,
                     prompt: string.Template, chat: ChatFn, package: str = "", topic_block: str = "",
                     literature_block: str = "", example_block: str = "") -> tuple[str | None, str]:
    """ONE repair of one function in ``source`` (a module's text), alone: ``(new module text, "")`` or ``(None, why)``.
    The request holds that function, its equation and the problem, not the script."""
    node = _find(source, name)
    current = ast.get_source_segment(source, node) if node is not None else signature
    reply = await chat(prompt.substitute(equation_block=equation_block, function_block=current or signature,
                                         problem=problem, related_block="(none)", package=package or "the package",
                                         topic_block=topic_block or "(the topic is not available)",
                                         literature_block=literature_block, example_block=example_block))
    return replace_function(source, name, reply_code(reply), signature=signature, equation=equation)
