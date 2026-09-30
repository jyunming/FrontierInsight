"""The quest's ``code/`` as a small research tool: the default shape of a simulation's code.

Two scripts (``simulate.py``, ``experiment.py``; :mod:`core.split_run`, :mod:`core.trial_runner`) are what FI runs. By
default the code is also laid out the way a small research tool is:

- ``code/<package>/``: the model's equations and nothing else (no grid values, sizes, seeds or file paths), each function
  carrying its equation's label (``# E1``) from the plan's *The model behind the numbers*;
- ``code/simulate.py``: still the file FI imports and calls (``run_cell`` / ``run_trial`` / ``oracle``), now the scenario
  around the package: it reads one setting and computes it with the package's functions;
- ``code/experiment.py``: the analysis, as before;
- ``code/tests/test_oracles.py``: the plan's checks against known answers as unit tests, written by FI from the plan
  (the values they must agree with come from the plan, not from the code);
- ``code/METHODS.md``: each equation of the model and the function that computes it, written by FI from the labels;
- ``code/run.py`` (:mod:`core.code_project`): the sweep over every setting and the one command that runs it all.

The model writes the package and the scenario; FI writes the tests, the equation list and ``run.py``. The extra cost is
estimated at plan time (:func:`estimate`) and shown in ``plan.md`` (:func:`plan_lines`); over the limit set in
``execution.code_package_max_extra_lines`` / ``code_package_max_extra_calls`` the quest keeps the two scripts alone and
says so (:func:`decide`). After the code is written the engine checks the layout (:func:`check`): what is missing is a
warning, and a stop only under ``rigor_profile: research``.
"""

from __future__ import annotations

import ast
import io
import json
import keyword
import re
import sys
import tokenize
from pathlib import Path
from typing import Any

from . import oracle_check as _oracle

PACKAGE = "package"
SINGLE = "single"
RECORD = Path(".fi") / "code_layout.json"
MODEL_NAME = "model.py"
TESTS_DIR = "tests"
TEST_NAME = "test_oracles.py"
TEST_PATH = f"{TESTS_DIR}/{TEST_NAME}"
METHODS_NAME = "METHODS.md"
HEADING = "How the code will be laid out"

# What a package name must not be: the scripts FI runs beside it, and names a reader would take for something else.
_TAKEN = {"simulate", "experiment", "run", "submit", "tests", "test", "raw", "figures", "fi", "core", "code", "model"}
_FILE_MARKER = re.compile(r"^\s*#\s*file\s*:\s*([\w.\-/\\]+)\s*$", re.IGNORECASE)

# The estimate's parts, in lines of code. What the model writes beyond two scripts: the package's own file, the imports
# and docstrings of the model module, and the calls simulate.py makes into it (the equations themselves move, they are
# not written twice). What FI writes: the tests (a header, then a few lines per check) and the equation list.
_MODEL_BASE_LINES = 20
_MODEL_LINES_PER_EQUATION = 5
_TEST_BASE_LINES = 45
_TEST_LINES_PER_CHECK = 4
_METHODS_BASE_LINES = 12
# Files beyond two scripts: <package>/__init__.py, <package>/model.py, tests/test_oracles.py, METHODS.md.
_EXTRA_FILES = 4


# --- the decision, at plan time --------------------------------------------------------------------------------------


def package_name(title: str) -> str:
    """A package name from the quest's title: lower case, words joined by ``_``, at most 30 characters, never a name
    Python or the scripts beside it already use. ``study_model`` when the title gives nothing usable."""
    words = re.findall(r"[a-z0-9]+", str(title or "").lower())
    name = ""
    for word in words:
        joined = f"{name}_{word}" if name else word
        if len(joined) > 30:
            break
        name = joined
    if name and name[0].isdigit():
        name = f"study_{name}"
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    if not name or not name.isidentifier() or keyword.iskeyword(name) or name in _TAKEN or name in stdlib:
        return "study_model"
    return name


def estimate(protocol: dict[str, Any] | None, *, max_iterations: int) -> dict[str, int]:
    """How much more the package layout costs than two scripts, from the plan: files, lines of code (the model's part and
    FI's part) and requests to the model. The requests are an upper bound: one per time the code is written (the first
    time and once per revision, ``engine.max_iterations``), made only when a reply leaves the package out."""
    equations = len(_oracle.generating_equations(protocol))
    checks = len([o for o in _oracle.declared(protocol) if _oracle.limit_of(o)[1] is not None])
    by_model = _MODEL_BASE_LINES + _MODEL_LINES_PER_EQUATION * equations
    by_fi = _TEST_BASE_LINES + _TEST_LINES_PER_CHECK * checks + _METHODS_BASE_LINES + equations
    return {
        "extra_files": _EXTRA_FILES,
        "extra_lines": by_model + by_fi,
        "extra_lines_by_model": by_model,
        "extra_lines_by_fi": by_fi,
        "extra_calls": 1 + max(0, int(max_iterations or 0)),
        "equations": equations,
        "checks": checks,
    }


def decide(protocol: dict[str, Any] | None, *, enabled: bool, max_extra_lines: int, max_extra_calls: int,
           max_iterations: int, package: str) -> dict[str, Any]:
    """The shape of this quest's code: ``package`` (the default) or ``single`` (two scripts alone), with the estimate and,
    for ``single``, why, in plain words."""
    cost = estimate(protocol, max_iterations=max_iterations)
    decision: dict[str, Any] = {"shape": PACKAGE, "package": package, "estimate": cost,
                                "limits": {"extra_lines": int(max_extra_lines), "extra_calls": int(max_extra_calls)},
                                "reason": ""}
    if not enabled:
        decision.update(shape=SINGLE, reason="execution.code_package is off")
        return decision
    over = []
    if cost["extra_lines"] > max_extra_lines:
        over.append(f"about {cost['extra_lines']} more lines of code, over the limit of {max_extra_lines} "
                    "(execution.code_package_max_extra_lines)")
    if cost["extra_calls"] > max_extra_calls:
        over.append(f"up to {cost['extra_calls']} more requests to the model, over the limit of {max_extra_calls} "
                    "(execution.code_package_max_extra_calls)")
    if over:
        decision.update(shape=SINGLE, reason="the package layout would need " + " and ".join(over))
    return decision


def summary(decision: dict[str, Any]) -> str:
    """One sentence for run.log."""
    cost = decision.get("estimate") or {}
    pkg = decision.get("package") or "the package"
    if decision.get("shape") == PACKAGE:
        return (f"the code is laid out as a small research tool (the model's equations in code/{pkg}/, unit tests, "
                f"an equation list): about {cost.get('extra_lines')} more lines of code than two scripts and at most "
                f"{cost.get('extra_calls')} more requests to the model")
    return (f"the code keeps two scripts (simulate.py and experiment.py) instead of the research-tool layout: "
            f"{decision.get('reason') or 'it is off'}")


def plan_lines(decision: dict[str, Any] | None) -> list[str]:
    """The plan.md section saying how the code will be laid out and what that costs. Nothing here is read back."""
    if not decision:
        return []
    cost = decision.get("estimate") or {}
    limits = decision.get("limits") or {}
    pkg = decision.get("package") or "the package"
    lines = [f"## {HEADING}", ""]
    what = (f"about {cost.get('extra_files')} more files and about {cost.get('extra_lines')} more lines of code than two "
            f"scripts ({cost.get('extra_lines_by_model')} written by the model, {cost.get('extra_lines_by_fi')} by FI), "
            f"and at most {cost.get('extra_calls')} more requests to the model (one each time the code is written, "
            "only if a reply leaves the package out)")
    if decision.get("shape") == PACKAGE:
        lines += [
            f"The code in `code/` will be a small research tool: the model's equations in the package `code/{pkg}/` "
            "(no scenario values there), `simulate.py` running one setting with it (the function FI calls), "
            "`experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, "
            "`METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.",
            "",
            f"This costs {what}. The limit is {limits.get('extra_lines')} lines and {limits.get('extra_calls')} "
            "requests (`execution.code_package_max_extra_lines`, `execution.code_package_max_extra_calls`).",
            "",
        ]
    else:
        lines += [
            f"The code will keep two scripts, `simulate.py` and `experiment.py`, not the research-tool layout: "
            f"{decision.get('reason') or 'it is off'}.",
            "",
        ]
        if "limit" in str(decision.get("reason") or ""):
            lines += [f"The research-tool layout would cost {what}. Raise the limit to get it.", ""]
    return lines


def load(quest_root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save(quest_root: Path, decision: dict[str, Any]) -> None:
    path = Path(quest_root) / RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision, indent=1), encoding="utf-8")
    except OSError:
        pass


# --- what the code-writing step is asked for, and what it returns ------------------------------------------------------


def prompt_block(package: str, equations: list[str]) -> str:
    """The layout the code-writing prompts ask for, after the two-script contract."""
    labels = ", ".join(equations) if equations else "each equation of the plan's model"
    return f"""
THE CODE AS A SMALL RESEARCH TOOL. Besides simulate.py and experiment.py, write the model as a package, each file its own
fenced Python block whose first line is `# file: <path>`:
- `# file: {package}/__init__.py`: one line saying what the package computes (it may import the model's functions).
- `# file: {package}/{MODEL_NAME}`: the model's equations and nothing else: one function per equation (or per small group),
  every parameter an argument. No scenario values in the package (no grid values, sizes, seeds, thresholds or file paths)
  and no loop over settings. Put each equation's id in a comment (`# E1`) on or above the function that computes it:
  {labels}.
- simulate.py keeps run_trial / run_cell (and oracle) exactly as the contract above says: it reads the setting from
  `cell` and computes it by calling the package (`from {package} import model`). The scenario is in simulate.py, the
  mathematics in the package.
Do not write tests, a README or a command-line entry: FI writes the unit tests from the plan's checks
(`tests/{TEST_NAME}`), the equation list (`{METHODS_NAME}`) and `run.py` itself.
"""


def reminder(package: str) -> str:
    return (f"\n\nREMINDER: the reply did not hold the package. Give every file again, each in its own fenced block: "
            f"`# file: simulate.py`, `# file: experiment.py`, `# file: {package}/__init__.py` and "
            f"`# file: {package}/{MODEL_NAME}`.\n")


def reply_files(text: str, fence: re.Pattern[str], package: str) -> dict[str, str]:
    """The package's files in a code-writing reply, ``{"<package>/model.py": text}``; only Python files directly inside
    the package are taken."""
    out: dict[str, str] = {}
    for match in fence.finditer(text or ""):
        block = match.group(1).strip("\n")
        lines = block.splitlines()
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first is None:
            continue
        marker = _FILE_MARKER.match(lines[first])
        if not marker:
            continue
        rel = _package_rel(marker.group(1), package)
        if rel is None:
            continue
        out[rel] = "\n".join(lines[first + 1:]).strip("\n") + "\n"
    return out


def _package_rel(path: str, package: str) -> str | None:
    """``"<package>/<module>.py"`` for a path naming a Python file directly inside the package, else ``None``."""
    rel = str(path or "").strip().replace("\\", "/").strip("/")
    if rel.startswith("code/"):
        rel = rel[len("code/"):]
    parts = rel.split("/")
    if len(parts) != 2 or parts[0] != package or not parts[1].endswith(".py") or not parts[1][:-3].isidentifier():
        return None
    return rel


def repair_files(value: Any, package: str) -> dict[str, str]:
    """The package files a repair reply gives back (its JSON ``package_files``: ``{"<package>/model.py": text}``); only
    Python files directly inside the package that parse are taken."""
    out: dict[str, str] = {}
    if not isinstance(value, dict):
        return out
    for path, text in value.items():
        rel = _package_rel(str(path), package)
        if rel is None or not isinstance(text, str) or not text.strip() or not _parses(text):
            continue
        out[rel] = text.strip("\n") + "\n"
    return out


def _top_functions(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def dropped_functions(old: str, new: str) -> list[str]:
    """The functions (and classes) ``old`` defines at its top level that ``new`` no longer does: a file given back with
    some of them missing is a part of the file, not the whole of it."""
    return sorted(_top_functions(old) - _top_functions(new))


def sources_block(sources: dict[str, str]) -> str:
    """The package's files as the prompts show them."""
    return "\n\n".join(f"### code/{rel}\n```python\n{text}\n```" for rel, text in sorted(sources.items()))


def repair_note(package: str, sources: dict[str, str]) -> str:
    """What a repair of simulate.py is told about the package it imports: its files, and how to fix one of them."""
    if not sources:
        return ""
    return (
        f"\nTHE MODEL'S PACKAGE: simulate.py computes with the package code/{package}/, which holds the model's equations. "
        "Its files are below. If the fault is in an equation, fix it there: add to your JSON "
        f"`\"package_files\": {{\"{package}/{MODEL_NAME}\": \"<the whole corrected file>\"}}` beside `code` (which is "
        "still the whole of simulate.py). Keep the equations in the package (do not copy them into simulate.py) and keep "
        "every equation label (`# E1`). Leave `package_files` out when the package is right.\n\n"
        + sources_block(sources) + "\n"
    )


def extend_note(package: str) -> str:
    """What an extension is told about the package shown beside the scripts."""
    return (
        f"The model's package code/{package}/ is shown above too. Give each of its files back exactly as it is unless "
        "what is asked needs a change to an equation; a package file you change comes back whole (every function it has "
        "now, plus what you add), never only the new part."
    )


def complete(files: dict[str, str], package: str) -> bool:
    return f"{package}/{MODEL_NAME}" in files and bool(files[f"{package}/{MODEL_NAME}"].strip())


def write_package(code_dir: Path, files: dict[str, str], *, same: Any = None) -> list[str]:
    """Write the package's files into ``code/``; a file that came back the same (``same(old, new)``) is left as it is, so
    the trials already run against it stay in use. Returns the files written."""
    written = []
    for rel, text in sorted(files.items()):
        path = Path(code_dir) / rel
        old = path.read_text(encoding="utf-8") if path.is_file() else None
        if old is not None and (same(old, text) if same else old == text):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(rel)
    if files:
        init = Path(code_dir) / next(iter(files)).split("/")[0] / "__init__.py"
        if not init.is_file():
            init.write_text('"""The model\'s equations."""\n', encoding="utf-8")
            written.append(init.relative_to(code_dir).as_posix())
    return written


# --- reading the package back ------------------------------------------------------------------------------------------


def package_dirs(code_dir: Path) -> list[Path]:
    """The Python packages in ``code/`` (a folder with ``__init__.py``), not the tests."""
    code_dir = Path(code_dir)
    if not code_dir.is_dir():
        return []
    return sorted(p for p in code_dir.iterdir()
                  if p.is_dir() and (p / "__init__.py").is_file() and p.name not in (TESTS_DIR, "__pycache__"))


def package_sources(code_dir: Path) -> dict[str, str]:
    """``{"<package>/model.py": text}`` for every Python file of the packages in ``code/``."""
    out: dict[str, str] = {}
    for pkg in package_dirs(code_dir):
        for path in sorted(pkg.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                out[path.relative_to(code_dir).as_posix()] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    return out


def _functions(tree: ast.AST) -> list[tuple[int, int, str]]:
    """``(first line, last line, qualified name)`` of every function, decorators included."""
    out: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    out.append((start, getattr(child, "end_lineno", child.lineno) or child.lineno, name))
                walk(child, f"{name}.")

    walk(tree, "")
    return out


def _labels(source: str) -> list[tuple[int, str, str | None]]:
    """``(line, text, function it names)`` for every comment and function docstring of ``source``: a comment inside a
    function, or on the lines right above one, names that function; a docstring names its own."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    functions = _functions(tree)
    found: list[tuple[int, str, str | None]] = []
    try:
        comments = [(tok.start[0], tok.string) for tok in tokenize.generate_tokens(io.StringIO(source).readline)
                    if tok.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        comments = []
    for line, text in comments:
        inside = [f for f in functions if f[0] <= line <= f[1]]
        if inside:
            found.append((line, text, max(inside, key=lambda f: f[0])[2]))
            continue
        below = [f for f in functions if 0 < f[0] - line <= 3]
        found.append((line, text, min(below, key=lambda f: f[0])[2] if below else None))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                name = next((f[2] for f in functions if f[0] <= node.lineno <= f[1] and f[2].split(".")[-1] == node.name),
                            node.name)
                found.append((node.lineno, doc, name))
    return found


def equation_map(protocol: dict[str, Any] | None, sources: dict[str, str]) -> list[dict[str, Any]]:
    """For each ``generates`` equation of the plan: where the package computes it, from its label. ``file`` and
    ``function`` are ``None`` when no function of the package carries the label."""
    equations = {str(e.get("id") or "").strip(): e for e in
                 (((protocol or {}).get("model") or {}).get("equations") or []) if isinstance(e, dict)}
    labelled = {rel: _labels(text) for rel, text in sorted(sources.items())}
    rows = []
    for eid in _oracle.generating_equations(protocol):
        where = next(((rel, fn, line) for rel, found in labelled.items() for line, text, fn in found
                      if fn and _oracle._labelled_in(eid, text)), None)
        rows.append({"id": eid, "formula": str((equations.get(eid) or {}).get("formula") or ""),
                     "file": where[0] if where else None, "function": where[1] if where else None,
                     "line": where[2] if where else None})
    return rows


# --- what FI writes itself -------------------------------------------------------------------------------------------


def methods_text(protocol: dict[str, Any] | None, rows: list[dict[str, Any]], package: str) -> str:
    """``METHODS.md``: the model and the function that computes each of its equations."""
    model = (protocol or {}).get("model")
    summary_text = model.get("summary") if isinstance(model, dict) else model if isinstance(model, str) else ""
    lines = ["# Methods: where each equation is computed", "",
             "Written by Frontier Insight from the plan's model and the equation labels in the code (`# E1`). It follows "
             "the code each time the code changes.", ""]
    if summary_text:
        lines += [f"**The model:** {' '.join(str(summary_text).split())}", ""]
    if rows:
        lines += ["| Equation | What it says | Where the code computes it |", "|---|---|---|"]
        for row in rows:
            formula = " ".join(row["formula"].split()).replace("|", "\\|") or "(not written)"
            where = (f"`{row['file']}`, `{row['function']}()` (line {row['line']})" if row["function"]
                     else f"not labelled in `{package}/`")
            lines.append(f"| {row['id']} | {formula} | {where} |")
        lines.append("")
    else:
        lines += ["The plan's model lists no equation that the simulation computes its data with.", ""]
    lines += ["## How the files fit together", "",
              f"- `{package}/`: the model's equations, one function each; no scenario values.",
              "- `simulate.py`: one setting of the study, computed with the package (the function Frontier Insight calls).",
              "- `experiment.py`: reads the results of every setting, computes the numbers and draws the figures.",
              f"- `{TEST_PATH}`: the plan's checks against known answers, as unit tests.",
              "- `run.py`: runs every setting, then the analysis (`python run.py`).", ""]
    return "\n".join(lines)


def _test_name(name: str, taken: set[str]) -> str:
    base = "test_" + ("_".join(re.findall(r"[a-z0-9]+", name.lower())) or "check")[:60]
    out, n = base, 2
    while out in taken:
        out, n = f"{base}_{n}", n + 1
    taken.add(out)
    return out


_TEST_HEADER = '''"""The plan's checks against known answers, as unit tests. Written by Frontier Insight from the plan; it follows the
plan each time the code changes.

    python -m pytest tests          (or, without pytest: python tests/{test_name})

Each test runs the simulation on the check's small case the way Frontier Insight does before the main run, and compares
the number with the value the plan fixed (and the tolerance it allows).
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("FI_THRESHOLDS", {thresholds!r})

import simulate  # noqa: E402

CHECKS = json.loads({checks!r})


def _seed(case):
    key = ",".join(f"{{axis}}={{repr(v) if isinstance(v, float) and v.is_integer() else v}}" for axis, v in case.items())
    base = int(os.environ.get("FI_REPLICATE_SEED") or 0)
    return int(hashlib.sha256(f"{{base}}|{{key}}|0".encode("utf-8")).hexdigest()[:8], 16)


def _measure(check):
    if check["case"] is None:
        return simulate.oracle()[check["name"]]
    case = dict(check["case"])
    if hasattr(simulate, "run_trial"):
        return simulate.run_trial(case, 0, _seed(case))[check["measure"]]
    return simulate.run_cell(case)[check["measure"]]


def _check(index):
    check = CHECKS[index]
    value = float(_measure(check))
    assert math.isfinite(value) and abs(value - check["expected"]) <= check["limit"], (
        f"{{check['name']}}: measured {{value}}, the plan expects {{check['expected']}} within {{check['limit']}}")
'''


def oracle_tests(protocol: dict[str, Any] | None) -> str | None:
    """``tests/test_oracles.py``: one test per check of the plan that has a number to agree with; ``None`` when there is
    none."""
    checks, names = [], []
    taken: set[str] = set()
    for oracle in _oracle.declared(protocol):
        expected, limit, _mode = _oracle.limit_of(oracle)
        if expected is None or limit is None:
            continue
        own = _oracle.case_of(oracle)
        name = str(oracle["name"]).strip()
        checks.append({"name": name, "case": own[0] if own else None, "measure": own[1] if own else None,
                       "expected": expected, "limit": limit})
        names.append(_test_name(name, taken))
    if not checks:
        return None
    thresholds = (protocol or {}).get("thresholds")
    text = _TEST_HEADER.format(test_name=TEST_NAME,
                               thresholds=json.dumps(thresholds if isinstance(thresholds, dict) else {}, sort_keys=True),
                               checks=json.dumps(checks, default=str))
    for i, (name, check) in enumerate(zip(names, checks)):
        text += f"\n\ndef {name}():\n    \"\"\"{check['name']}\"\"\"\n    _check({i})\n"
    text += ("\n\nif __name__ == \"__main__\":\n    failed = 0\n    for name, fn in [" +
             ", ".join(f"({n!r}, {n})" for n in names) + "]:\n"
             "        try:\n            fn()\n            print(\"ok  \", name)\n"
             "        except Exception as e:  # noqa: BLE001\n            failed += 1\n            print(\"FAIL\", name, e)\n"
             "    sys.exit(1 if failed else 0)\n")
    return text


def project_files(code_dir: Path, protocol: dict[str, Any] | None, package: str) -> dict[str, str]:
    """What FI writes into ``code/`` for the package layout: the equation list and the checks as unit tests."""
    rows = equation_map(protocol, package_sources(code_dir))
    files = {METHODS_NAME: methods_text(protocol, rows, package)}
    tests = oracle_tests(protocol)
    if tests:
        files[TEST_PATH] = tests
    return files


def readme_lines(package: str) -> list[str]:
    return [f"- `{package}/`: the model's equations, one function each (no scenario values); `simulate.py` uses it.",
            f"- `{TEST_PATH}`: the plan's checks against known answers, as unit tests (`python -m pytest tests`).",
            f"- `{METHODS_NAME}`: which function computes each equation of the model."]


# --- the check after the code is written -------------------------------------------------------------------------------


def _imports(source: str, package: str) -> bool:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == package for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] == package:
            return True
    return False


def check(code_dir: Path, protocol: dict[str, Any] | None, package: str) -> list[str]:
    """What the research-tool layout is missing, one plain sentence each (empty when nothing is): the package and its
    use by simulate.py, the unit tests, and the equation list with a function for every equation."""
    code_dir = Path(code_dir)
    problems: list[str] = []
    pkg = code_dir / package
    sources = {rel: text for rel, text in package_sources(code_dir).items() if rel.startswith(f"{package}/")}
    has_function = any(_functions(ast.parse(text)) for text in sources.values() if _parses(text))
    if not (pkg / "__init__.py").is_file() or not has_function:
        problems.append(f"the model's package code/{package}/ is missing (or holds no function), so the equations are "
                        "not kept apart from the scenario")
    else:
        try:
            simulate = (code_dir / "simulate.py").read_text(encoding="utf-8")
        except OSError:
            simulate = ""
        if not _imports(simulate, package):
            problems.append(f"code/simulate.py does not use the package code/{package}/, so the equations it runs are "
                            "not the ones in the package")
    # FI writes the unit tests from the plan's checks with a number to agree with; a plan with none has none to write,
    # and that is not something the code is missing.
    tests = [p for p in (code_dir / TESTS_DIR).glob("test_*.py")] if (code_dir / TESTS_DIR).is_dir() else []
    if oracle_tests(protocol) is not None and not any(
            re.search(r"^def test_", p.read_text(encoding="utf-8", errors="replace"), re.MULTILINE) for p in tests):
        problems.append(f"there are no unit tests in code/{TESTS_DIR}/ for the plan's checks")
    if not (code_dir / METHODS_NAME).is_file():
        problems.append(f"code/{METHODS_NAME}, which says which function computes each equation, is missing")
    unmapped = [row["id"] for row in equation_map(protocol, sources) if not row["function"]]
    if unmapped:
        problems.append(f"equation{'s' if len(unmapped) > 1 else ''} {', '.join(unmapped)} of the model "
                        f"{'are' if len(unmapped) > 1 else 'is'} not labelled on a function of code/{package}/ "
                        f"(a comment such as `# {unmapped[0]}` on or above it), so {METHODS_NAME} cannot say where "
                        f"{'they are' if len(unmapped) > 1 else 'it is'} computed")
    return problems


def _parses(text: str) -> bool:
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return True
