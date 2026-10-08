"""A different model reads the generated code against the plan's required outputs, before the study runs.

A script can run to the end, print a plausible ``RESULT_JSON`` and still leave out a quantity the plan requires (a
dependent variable, a metric, a planned figure, a swept parameter, the equation a function was meant to compute), or
compute it wrongly. The model that wrote the code reads its own code as the plan says; nothing else checks it. Here,
after the code is written (and the model's functions are checked one at a time) and before the equation tests and the
oracle gate, the model named for ``code_review`` (while no model is set for it, the one named for ``oracle_review``, so it
is the other provider when one is configured) reads the plan's required outputs against the scripts and says, per
requirement, whether it is **implemented**, **missing** or **implemented differently**, with the file and line.

The same reading also checks the LOGIC against the plan (core/review_pairs.py: a reviewer finds problems, each with where,
what, why, how to fix and how serious): that each ``# E<n>`` function matches its equation (normalisations, units,
factors), that a statistic is computed as the plan defines it, that a quantity the plan says depends on a setting uses that
setting, that a step the plan's model names is not skipped, that nothing reported is a constant, and that a check's case
is the plan's. Its findings go to a repair request of their own, each naming the place and the fault, never a value;
whatever value of the plan's checks a reviewer quotes is replaced in the findings (:func:`core.review_pairs.without_values`). A finding the run's own results
confirm (a quantity the reviewer calls constant that the results hold at one value, :func:`constant_in`) about a headline quantity
ends in the honest stop after the run; every other open finding is said in the paper's limitations as unresolved.

* A reader that is the model that wrote the code is not a second reader: nothing is asked, and ``.fi/code_review.json``
  and ``run.log`` say that the code was not read by another model.
* What is missing or different goes to its OWN repair, not the repair of a crashed script: the code has not crashed, it
  does not do what the plan requires. A request of its own (``agents/code_review_repair.md``) holds the topic, the plan's
  equations (never an expected value), the findings IN FULL (each with where, what, why, fix and severity; never a
  value) and the file the findings name (:func:`files_for`: the model's package, ``simulate.py`` or ``experiment.py``),
  in batches of at most :data:`BATCH` findings, at most :data:`MAX_SEND_BACKS` rounds and :data:`MAX_REPAIR_CALLS`
  requests in a quest. After a round the changed version is read again (against the findings still open, and anything
  new); when the open high-severity findings did not drop, repairing stops (no oscillation).
* One reading per code version (:func:`version`), at most :data:`MAX_REVIEWS` in a quest, and the last of them is
  reserved for the version that will run the full study: an intermediate version never spends it. The final version is
  read, and the record says whether it was (``final_read``), the open findings per reading and whether the repairs
  converged. A version that a repair of the run itself (a crash, a result the protocol's contract refuses) changed after
  that reading is read again before its results are used, within :data:`RUN_REPAIR_READS` more readings.
* What is still open afterwards is said in the paper's limitations (:func:`write_note`). A high-severity finding about
  a headline quantity (the first dependent variable, the metric the precision target is for) that is still open in the
  final version ends in the honest stop (``stuck_no_findings``, no paper) after the run, unless the run's own results
  contradict it (a quantity the reviewer calls missing that the results hold, one it calls constant that they show
  varying); that disagreement is recorded and the paper is not told it was not computed. A reviewer's word alone never
  stops a quest before the run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import oracle_check as _oracle
from . import review_pairs as _pairs

RECORD = Path(".fi") / "code_review.json"
#: Reviews of the code in a whole quest (one per code version; the last is reserved for the version that will run), the
#: rounds of repair of what a review found, the findings in one repair request, and all repair requests in a quest.
MAX_REVIEWS = 3
RESERVED_FOR_FINAL = 1
#: Further readings for the version that runs when a run that crashed or broke the protocol's contract was repaired by
#: ``execute_reflect`` AFTER the reserved reading was spent: that script is the one whose results are reported, so it is
#: read too (at most one per repair, and ``engine.exec_reflect_max_iterations`` bounds the repairs; 3 by default).
RUN_REPAIR_READS = 3
MAX_SEND_BACKS = 2
BATCH = 6
MAX_REPAIR_CALLS = 8
#: How much code the reader is shown (whole files; the rest is said not to be shown).
_MAX_CODE_CHARS = 60000
IMPLEMENTED, MISSING, DIFFERENT = "implemented", "missing", "different"
_STATUS_WORDS = {
    "implemented": IMPLEMENTED, "yes": IMPLEMENTED, "ok": IMPLEMENTED, "present": IMPLEMENTED, "done": IMPLEMENTED,
    "missing": MISSING, "absent": MISSING, "no": MISSING, "not implemented": MISSING,
    "different": DIFFERENT, "differently": DIFFERENT, "implemented differently": DIFFERENT, "differs": DIFFERENT,
}


def _text(value: Any, limit: int = 300) -> str:
    return " ".join(str(value if value is not None else "").replace("`", "'").split())[:limit]


#: The kind of a requirement that asks the reader to confirm a ``# constant by design`` mark in the code.
MARK_KIND = "constant by design"


def constant_marks(code_dir: Path) -> list[Any]:
    """The valid ``# constant by design`` marks in the code (:func:`core.degenerate_results.design_marks`)."""
    from . import degenerate_results as _degenerate
    from . import grid_keys as _grid_keys

    return _degenerate.design_marks(_grid_keys.code_files(Path(code_dir)))


def requirements(design: dict[str, Any] | None, protocol: dict[str, Any] | None,
                 marks: list[Any] | None = None) -> list[dict[str, Any]]:
    """The plan's required outputs, as ``{"id", "kind", "text", "headline"}`` (R1, R2 ...): the dependent variables, the
    metrics, the planned figures, the swept parameters, the checks' measures and the equations the simulation computes
    its data with. Names and descriptions only, never a number the code must reproduce."""
    design = design if isinstance(design, dict) else {}
    protocol = protocol if isinstance(protocol, dict) else {}
    out: list[dict[str, Any]] = []

    def add(kind: str, text: str, headline: bool = False, name: str = "") -> None:
        text = _text(text)
        if text and not any(r["kind"] == kind and r["text"] == text for r in out):
            out.append({"id": f"R{len(out) + 1}", "kind": kind, "text": text, "headline": headline, "name": _text(name, 80)})

    precision = protocol.get("precision") if isinstance(protocol.get("precision"), dict) else {}
    headline_metric = _text(precision.get("metric")).lower()
    variables = design.get("variables") if isinstance(design.get("variables"), dict) else {}
    dependent = [d for d in (variables.get("dependent") if isinstance(variables.get("dependent"), list) else []) if _text(d)]
    for i, name in enumerate(dependent):
        add("dependent variable", f"the dependent variable {_text(name)} is computed and reported in the result", i == 0, str(name))
    for m in protocol.get("metrics") if isinstance(protocol.get("metrics"), list) else []:
        if isinstance(m, dict) and _text(m.get("id")):
            mid = _text(m["id"])
            est = _text(m.get("estimand"))
            add("metric", f"the result reports {mid}" + (f" ({est})" if est else ""),
                bool(headline_metric) and (mid.lower() == headline_metric or headline_metric in mid.lower()), mid)
    for fig in design.get("figures_planned") if isinstance(design.get("figures_planned"), list) else []:
        add("figure", f"the script draws the planned figure {Path(str(fig)).name}")
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    for name, values in grid.items():
        if isinstance(values, list) and len(values) > 1:
            add("sweep", f"the parameter {_text(name, 60)} is varied over all {len(values)} values of the plan's grid")
    for o in _oracle.declared(protocol):
        measure = _text(o.get("measure"), 160)
        if measure:
            add("check", f"the simulation returns what the measure '{measure}' of the check '{_text(o['name'], 60)}' uses")
    for eid in _oracle.generating_equations(protocol):
        add("equation", f"equation {eid} of the plan's model is computed in the code, in a function marked `# {eid}`")
    for mark in marks or []:
        add(MARK_KIND, mark.asks(), name=mark.key())
    return out


def sources_text(code_dir: Path) -> tuple[str, bool]:
    """``(the scripts with line numbers, whether some were left out)``: simulate.py, experiment.py and the model's
    package; whole files only, as many as fit."""
    from . import code_layout as _layout

    code_dir = Path(code_dir)
    files: dict[str, str] = {}
    for name in ("simulate.py", "experiment.py"):
        if (code_dir / name).is_file():
            files[name] = (code_dir / name).read_text(encoding="utf-8", errors="replace")
    files.update(_layout.package_sources(code_dir))
    parts: list[str] = []
    used, partial = 0, False
    for name, text in files.items():
        numbered = "\n".join(f"{i:4d}  {line}" for i, line in enumerate(text.splitlines(), start=1))
        block = f"### {name}\n```python\n{numbered}\n```\n"
        if used + len(block) > _MAX_CODE_CHARS and parts:
            partial = True
            continue
        parts.append(block)
        used += len(block)
    return "\n".join(parts) + ("\n(some files are not shown)" if partial else ""), partial


def version(code_dir: Path) -> str:
    """A short hash of the code the review reads: a different one is another code version."""
    from . import code_layout as _layout

    code_dir = Path(code_dir)
    h = hashlib.sha256()
    for name in ("simulate.py", "experiment.py"):
        if (code_dir / name).is_file():
            h.update(name.encode() + (code_dir / name).read_bytes())
    for rel, text in sorted(_layout.package_sources(code_dir).items()):
        h.update(rel.encode() + text.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def parse(reply: Any, reqs: list[dict[str, Any]], values: list[float] | None = None) -> list[dict[str, Any]] | None:
    """The reader's verdict per requirement the plan has, or ``None`` when the answer names none of them. A requirement it
    does not mention is not judged (and so is never repaired); an id the plan does not have is dropped."""
    items = reply.get("requirements") if isinstance(reply, dict) else None
    if not isinstance(items, list):
        return None
    known = {r["id"].upper(): r for r in reqs}
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        req = known.get(_text(item.get("id"), 20).upper())
        status = _STATUS_WORDS.get(_text(item.get("status"), 40).lower().rstrip("."))
        if req is None or status is None or any(o["id"] == req["id"] for o in out):
            continue
        line = item.get("line")
        out.append({"id": req["id"], "kind": req["kind"], "text": req["text"], "headline": req["headline"],
                    "name": req.get("name", ""), "status": status, "file": _text(item.get("file"), 80),
                    "line": line if isinstance(line, int) and not isinstance(line, bool) else None,
                    "note": _pairs.scrub(_text(item.get("note")), values)})
    return out or None


def problems(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [v for v in verdicts if v["status"] in (MISSING, DIFFERENT)]


#: How much of the plan the reader is shown for the logic checks.
_MAX_PLAN_CHARS = 12000
_WHERE_RE = re.compile(r"([\w./\-]+\.py)(?:\s*[:,]?\s*(?:lines?\s*)?(\d+))?", re.I)


def plan_view(design: dict[str, Any] | None, protocol: dict[str, Any] | None) -> str:
    """What the logic checks are made against, as JSON text: the plan's model (its equations as written), the method, the
    statistics it names, the grid, and the checks' cases and measures. Never an expected value, a tolerance or a worked
    example: the reader is asked what is wrong, not what a number should be."""
    design = design if isinstance(design, dict) else {}
    protocol = protocol if isinstance(protocol, dict) else {}
    model = protocol.get("model") if isinstance(protocol.get("model"), dict) else {}
    equations = [{k: e.get(k) for k in ("id", "formula", "role", "source") if e.get(k)}
                 for e in (model.get("equations") if isinstance(model.get("equations"), list) else []) if isinstance(e, dict)]
    metrics = [{k: m.get(k) for k in ("id", "estimand", "kind") if m.get(k)}
               for m in (protocol.get("metrics") if isinstance(protocol.get("metrics"), list) else []) if isinstance(m, dict)]
    precision = protocol.get("precision") if isinstance(protocol.get("precision"), dict) else {}
    variables = design.get("variables") if isinstance(design.get("variables"), dict) else {}
    view = {
        "model": _text(model.get("summary"), 600), "equations": equations,
        "method": _text(design.get("method"), 1500),
        "settings_varied": variables.get("independent") if isinstance(variables.get("independent"), list) else [],
        "grid": protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {},
        "metrics": metrics, "headline_metric": _text(precision.get("metric"), 80),
        "ci_method": _text(protocol.get("ci_method"), 200), "seed_policy": _text(protocol.get("seed_policy"), 200),
        "checks": [{"name": _text(o.get("name"), 60), "kind": _text(o.get("kind"), 30), "check": _text(o.get("check"), 200),
                    "case": o.get("case"), "measure": _text(o.get("measure"), 160)} for o in _oracle.declared(protocol)],
    }
    values = _pairs.plan_values(protocol)

    def clean(node: Any) -> Any:
        """The free text the reader is shown, without the plan's values (it could echo them into a finding)."""
        if isinstance(node, str):
            return _pairs.scrub(node, values)
        if isinstance(node, list):
            return [clean(x) for x in node]
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items()}
        return node

    text = json.dumps(clean({k: v for k, v in view.items() if v not in ("", [], {}, None)}), indent=2, default=str)
    return text if len(text) <= _MAX_PLAN_CHARS else text[:_MAX_PLAN_CHARS] + "\n(the rest of the plan is not shown)"


def parse_logic(reply: Any, reqs: list[dict[str, Any]], values: list[float] | None = None) -> list[dict[str, Any]]:
    """The problems the reader found in how the code computes what the plan says (``findings`` of its reply), as
    problems like a requirement's: ``{"id": "L1", "kind": "logic", "text", "status": "different", "file", "line", "note",
    "fix", "severity", "name" (the quantity), "constant", "headline"}``. A value of the plan's checks or equation
    examples a reader quoted is replaced (:func:`core.review_pairs.without_values`); the other numbers stay."""
    if not isinstance(reply, dict):
        return []
    headline = {_norm(r["name"]) for r in reqs if r.get("headline") and r.get("name")}
    out: list[dict[str, Any]] = []
    for f in _pairs.findings(reply.get("findings"), limit=12):
        f = _pairs.without_values(f, values)
        match = _WHERE_RE.search(f["where"])
        quantity = f["quantity"]
        out.append({"id": f"L{len(out) + 1}", "kind": "logic", "text": f["what"], "status": DIFFERENT,
                    "file": match.group(1).replace("\\", "/").split("/")[-1] if match else "",
                    "line": int(match.group(2)) if match and match.group(2) else None,
                    "note": f["why"], "fix": f["fix"], "severity": f["severity"], "name": quantity,
                    "constant": bool(f.get("constant")), "where": f["where"],
                    "headline": bool(quantity) and _norm(quantity) in headline})
    return out


def merge_logic(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``old`` with the findings of ``new`` that say something it does not (one finding is one place and one fault)."""
    out = list(old)
    seen = {(v["where"], v["text"]) for v in out}
    for v in new:
        if (v["where"], v["text"]) not in seen:
            out.append({**v, "id": f"L{len(out) + 1}"})
            seen.add((v["where"], v["text"]))
    return out


def reply_said_nothing(verdicts: list[dict[str, Any]] | None, logic: list[dict[str, Any]], listed: list[str]) -> bool:
    """Whether the reader's answers add up to no review: no requirement is missing or different, no problem of logic is
    named, nothing it checked is listed, and no requirement it calls implemented says where (file or line) or why."""
    shown = any(v["file"] or v["line"] or v["note"] for v in verdicts or [] if v["status"] == IMPLEMENTED)
    return _pairs.said_nothing(logic, listed, other_problem=bool(problems(verdicts or [])), evidence=shown)


_ORDER = {"high": 0, "medium": 1, "low": 2}
_SIMULATION_KINDS = ("sweep", "check", "equation", "logic")


def severity_of(v: dict[str, Any]) -> str:
    """How serious an open problem is: a finding says so itself; a requirement of the plan that is missing or different is
    high when it is about a headline quantity, else medium."""
    if v.get("kind") == "logic":
        return v.get("severity") if v.get("severity") in _ORDER else "medium"
    return "high" if v.get("headline") else "medium"


def open_counts(problems: list[dict[str, Any]]) -> tuple[int, int]:
    """``(open findings, open high-severity findings)``."""
    return len(problems), sum(1 for v in problems if severity_of(v) == "high")


def files_for(problems: list[dict[str, Any]], code_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """The file each problem is to be repaired in, ``{"<path under code/>": [problems]}`` (the most serious first). The file
    a finding names wins (``model.py`` is the model's package, ``simulate.py``, ``experiment.py``); a finding that names
    none, or one that is not a file of the code, goes by what it is about: the simulation for a swept parameter, a check's
    measure, an equation or a fault of logic, the analysis for the rest."""
    from . import code_layout as _layout

    code_dir = Path(code_dir)
    scripts = [n for n in ("simulate.py", "experiment.py") if (code_dir / n).is_file()]
    candidates = [*scripts, *sorted(_layout.package_sources(code_dir))]
    by_name: dict[str, str] = {}
    for rel in candidates:
        by_name.setdefault(Path(rel).name.lower(), rel)
    fallback = {"simulation": "simulate.py" if "simulate.py" in scripts else (scripts[0] if scripts else ""),
                "analysis": "experiment.py" if "experiment.py" in scripts else (scripts[0] if scripts else "")}
    out: dict[str, list[dict[str, Any]]] = {}
    ranked = sorted(enumerate(problems), key=lambda iv: (_ORDER[severity_of(iv[1])], not iv[1].get("headline"), iv[0]))
    for _i, v in ranked:
        named = Path(str(v.get("file") or "").replace("\\", "/")).name.lower()
        rel = by_name.get(named) if named else None
        if rel is None:
            rel = fallback["simulation" if v.get("kind") in _SIMULATION_KINDS else "analysis"]
        if rel:
            out.setdefault(rel, []).append(v)
    return out


def batches(items: list[dict[str, Any]], size: int = BATCH) -> list[list[dict[str, Any]]]:
    return [items[i:i + size] for i in range(0, len(items), max(1, size))]


def findings_text(items: list[dict[str, Any]]) -> str:
    """The findings of one repair request, each IN FULL (where, what, why, fix, severity): nothing is cut."""
    lines = []
    for n, v in enumerate(items, 1):
        where = (v.get("where") or "") if v.get("kind") == "logic" else ""
        if not where and v.get("file"):
            where = f"{v['file']}" + (f":{v['line']}" if v.get("line") else "")
        head = f"{n}. [{severity_of(v)}] "
        if v.get("kind") == "logic":
            lines.append(head + f"Where: {where or 'not stated'}\n   What is wrong: {v['text']}\n"
                         f"   Why it is wrong against the plan: {v.get('note') or 'not stated'}\n"
                         f"   How to fix it: {v.get('fix') or 'not stated'}")
        else:
            what = "is missing from the code" if v["status"] == MISSING else "is implemented differently from the plan"
            lines.append(head + f"Where: {where or 'not stated'}\n   What is wrong: the plan's {v.get('kind')} \"{v['text']}\" {what}\n"
                         f"   Why it is wrong against the plan: {v.get('note') or 'the plan requires it'}\n"
                         "   How to fix it: make the code compute it as the plan states, and put it where the plan says")
    return "\n\n".join(lines)


def equations_view(protocol: dict[str, Any] | None, values: list[float] | None = None) -> str:
    """The plan's equations for a repair request as JSON text: id, formula, role and source; never an example or a value."""
    model = (protocol or {}).get("model") if isinstance(protocol, dict) else None
    items = (model.get("equations") if isinstance(model, dict) and isinstance(model.get("equations"), list) else [])
    rows = [{k: _pairs.scrub(_text(e.get(k), 400), values) for k in ("id", "formula", "role", "source") if e.get(k)}
            for e in items if isinstance(e, dict)]
    return json.dumps(rows, indent=1)


_FILE_LINE = re.compile(r"^\s*#\s*file\s*:\s*([\w.\-/\\]+)\s*$", re.IGNORECASE)
_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_files(reply: str, allowed: list[str]) -> dict[str, str]:
    """The files a repair reply gives back, ``{"<path under code/>": text}``: each a fenced Python block whose first line is
    ``# file: <name>``; only a file of ``allowed`` (by path, else by its name alone) and only one that parses."""
    import ast

    by_name = {Path(a).name.lower(): a for a in allowed}
    out: dict[str, str] = {}
    for m in _BLOCK.finditer(reply or ""):
        lines = m.group(1).strip("\n").splitlines()
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        marker = _FILE_LINE.match(lines[first]) if first is not None else None
        if not marker:
            continue
        name = marker.group(1).replace("\\", "/").strip("/")
        name = name[len("code/"):] if name.startswith("code/") else name
        rel = name if name in allowed else by_name.get(Path(name).name.lower())
        text = "\n".join(lines[first + 1:]).strip("\n") + "\n"
        if rel is None or not text.strip():
            continue
        try:
            ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        out[rel] = text
    return out


_MARK = "RESULT_JSON:"


def _starts_with_mark(node: ast.AST, names: dict[str, list[ast.AST]], depth: int = 0) -> bool:
    """Whether an expression is, statically, a text that begins with ``RESULT_JSON:`` (a literal, an f-string, a
    concatenation or ``%`` on one, a ``.format`` of one, or a name assigned one)."""
    if depth > 4:
        return False
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and node.value.lstrip().startswith(_MARK)
    if isinstance(node, ast.JoinedStr):
        first = node.values[0] if node.values else None
        return isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.lstrip().startswith(_MARK)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return _starts_with_mark(node.left, names, depth + 1)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _starts_with_mark(node.func.value, names, depth + 1)
    if isinstance(node, ast.Name):
        return any(_starts_with_mark(v, names, depth + 1) for v in names.get(node.id, []))
    return False


def prints_result_json(source: str) -> bool:
    """Whether the script has an executable statement that writes a line beginning ``RESULT_JSON:`` (``print(...)`` or a
    ``.write(...)``), the line FI reads the result from: not a comment, not a string that is never printed."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    names: dict[str, list[ast.AST]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.setdefault(t.id, []).append(n.value)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not n.args:
            continue
        f = n.func
        if (isinstance(f, ast.Name) and f.id == "print") or (isinstance(f, ast.Attribute) and f.attr == "write"):
            if _starts_with_mark(n.args[0], names):
                return True
    return False


def repair_problem(old: str, new: str, rel: str, *, equations: list[str], package: str = "") -> str:
    """Why a repaired file cannot replace the file (``""`` when it can): the repair changes what the findings name and
    nothing else, so it must keep every top-level function and constant the file had, every equation label, and (for the
    analysis) the line it prints its result on and its not importing the package."""
    from . import code_layout as _layout

    dropped = _layout.dropped_functions(old, new)
    if dropped:
        return "it dropped " + ", ".join(f"`{d}`" for d in dropped[:5]) + ", which the file had"
    for eid in equations:
        if _oracle._labelled_in(eid, old) and not _oracle._labelled_in(eid, new):
            return f"it lost the label `# {eid}`"
    if Path(rel).name == "experiment.py":
        if prints_result_json(old) and not prints_result_json(new):
            return "it no longer prints the RESULT_JSON line"
        if package and _layout.imports(new, package) and not _layout.imports(old, package):
            return f"it made experiment.py import the package `{package}`"
    return ""


#: What a record keeps from one code version to the next (the quest-wide counters and the history of the readings).
CARRIED = ("reviews_total", "send_backs", "repair_calls", "readings", "converged", "repaired", "read_version")


def carried(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in CARRIED if k in record}


def earlier_block(previous: list[dict[str, Any]]) -> str:
    """For the reading of a version a repair made: the findings still open in the version before, to be judged again in this
    one (``""`` for a first reading)."""
    if not previous:
        return ""
    lines = []
    for v in previous:
        where = v.get("where") or (v.get("file") or "") + (f":{v['line']}" if v.get("line") else "")
        lines.append(f"- [{severity_of(v)}] {v['text']}" + (f" ({where})" if where else "")
                     + (f". Why: {v['note']}" if v.get("note") else ""))
    return ("# Findings still open in the version before this one\n\nThe code above is the version after a repair. An earlier "
            "reading of the version before it left these findings open. For each, say whether it still holds in THIS version: "
            "if it does, list it again under `findings` (the same place, what, why and fix); if it does not, leave it out. "
            "Then judge the plan's requirements again, and add any new problem you find.\n\n" + "\n".join(lines) + "\n")


def may_repair(record: dict[str, Any]) -> str:
    """Why the problems the last reading found are not to be repaired now (``""`` when they are): the rounds or the requests
    are spent, no reading is left to read the repaired version, or the last repair did not reduce the open high-severity
    findings."""
    if int(record.get("send_backs") or 0) >= MAX_SEND_BACKS:
        return f"the {MAX_SEND_BACKS} rounds of repair allowed in a quest are spent"
    if int(record.get("repair_calls") or 0) >= MAX_REPAIR_CALLS:
        return f"the {MAX_REPAIR_CALLS} repair requests allowed in a quest are spent"
    if int(record.get("reviews_total") or 0) >= MAX_REVIEWS:
        return "no reading is left to read a repaired version"
    if record.get("converged") is False:
        return "the last repair did not reduce the open high-severity findings"
    return ""


def read_cap(*, final: bool, after_run_repair: bool = False) -> int:
    """The most readings a quest may have spent for a read to be made now: all but the reserved one for an intermediate
    version, all of them for the version about to run, and :data:`RUN_REPAIR_READS` more for a version a repair of the run
    itself changed (its results are the ones reported, so it is read)."""
    if not final:
        return MAX_REVIEWS - RESERVED_FOR_FINAL
    return MAX_REVIEWS + (RUN_REPAIR_READS if after_run_repair else 0)


def may_read(record: dict[str, Any], *, final: bool, after_run_repair: bool = False) -> bool:
    """Whether a reading is left (:func:`read_cap`)."""
    return int(record.get("reviews_total") or 0) < read_cap(final=final, after_run_repair=after_run_repair)


def note_reading(record: dict[str, Any], version_now: str, problems: list[dict[str, Any]]) -> None:
    """Record a reading: the open findings and the high-severity ones in it. When it is the reading of a version a repair
    made, it is compared with the reading before: the repairs have converged while each round reduced the open
    high-severity findings (or, with none open, the open findings), and not when one did not."""
    total, high = open_counts(problems)
    readings = [r for r in record.get("readings") or [] if isinstance(r, dict)]
    entry: dict[str, Any] = {"n": len(readings) + 1, "version": version_now, "open": total, "high": high}
    if record.get("repaired") and readings:
        before = readings[-1]
        improved = high < int(before.get("high") or 0) or (
            high == int(before.get("high") or 0) == 0 and total < int(before.get("open") or 0))
        entry["improved"] = improved
        record["converged"] = bool(improved) and record.get("converged") is not False
    record["repaired"] = False
    readings.append(entry)
    record["readings"] = readings


def final_read(quest_root: Path, code_dir: Path) -> bool:
    """Whether the version of the code that is there now is the one a second model read last."""
    record = load(quest_root)
    return bool(record.get("read_version")) and record.get("read_version") == version(code_dir)


def metrics_line(record: dict[str, Any]) -> str:
    """The readings in one sentence: the open findings in each (for run.log and the quality report)."""
    readings = [r for r in record.get("readings") or [] if isinstance(r, dict)]
    if not readings:
        return ""
    return ("open findings per reading: " + ", ".join(f"{r.get('open')} ({r.get('high')} high)" for r in readings)
            + ("; the repairs converged" if record.get("converged") else
               "; the repairs did not converge" if record.get("converged") is False else ""))


def directive(items: list[dict[str, Any]]) -> str:
    """What stands where a traceback would in the repair request: plain, naming each requirement, never a number."""
    lines = []
    for v in items:
        where = f" (see {v['file']}" + (f", line {v['line']}" if v["line"] else "") + ")" if v["file"] else ""
        if v.get("kind") == "logic":
            # A fault in how something is computed: where, what, why and how to fix, without any value.
            fix = f" Fix: {v['fix']}" if v.get("fix") else ""
            lines.append(f"- {v['text'].rstrip('.')}{where}" + (f". Why: {v['note']}" if v["note"] else "") + f".{fix}")
            continue
        what = "is missing from the code" if v["status"] == MISSING else "is implemented differently from the plan"
        lines.append(f"- {v['text']}: this {what}{where}" + (f". {v['note']}" if v["note"] else "."))
    return ("This code has NOT been run, and it has not failed: the account of a crash above does not apply. FI "
            "compared it with what the plan requires it to compute, and these requirements are not met:\n"
            + "\n".join(lines)
            + "\n\nChange exactly this: make the code do each of them as the plan states, and nothing else. Keep "
            "everything else unchanged: the same functions, the handling of FI_PILOT and FI_REPLICATE_SEED, and the "
            "final RESULT_JSON line (with the added quantities in it). Do not invent a value for a quantity: compute it.")


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


def not_computed(quest_root: Path, code_dir: Path) -> list[dict[str, Any]]:
    """The requirements the final version of the code still does not meet (empty when the record is for other code)."""
    record = load(quest_root)
    if not record.get("final") or record.get("version") != version(code_dir):
        return []
    return [v for v in record.get("problems") or [] if isinstance(v, dict)]


def headline_missing(quest_root: Path, code_dir: Path) -> list[str]:
    """The headline requirements the code is still said to miss. (A problem of logic is not a missing quantity: see
    :func:`headline_constant`.)"""
    return [v["text"] for v in not_computed(quest_root, code_dir)
            if v.get("headline") and not v.get("contradicted") and v.get("kind") != "logic"]


def headline_constant(quest_root: Path, code_dir: Path) -> list[str]:
    """The headline quantities a high-severity finding calls constant and the run's own results confirm are constant
    (:func:`resolve_logic`)."""
    return [v["text"] for v in not_computed(quest_root, code_dir) if v.get("confirmed") and v.get("headline")]


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def in_results(name: str, result: Any) -> bool:
    """Whether the run's own results hold a value under exactly ``name``, ignoring case, punctuation and underscores
    (``Quality_Factor`` and ``qualityFactor`` are one name; ``error_count`` is not ``error``): a number, or a
    non-empty list or mapping, never null or a flag."""
    want = _norm(name)
    if not want:
        return False

    def holds(value: Any) -> bool:
        if isinstance(value, bool) or value is None:
            return False
        if isinstance(value, (int, float)):
            return True
        return isinstance(value, (list, tuple, dict)) and len(value) > 0

    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            return any((_norm(k) == want and holds(v)) or walk(v) for k, v in node.items())
        if isinstance(node, (list, tuple)):
            return any(walk(v) for v in node if isinstance(v, (dict, list)))
        return False

    return walk(result)


def _holds_numbers(value: Any) -> bool:
    """Whether ``value`` is a number, or a list or mapping with a number somewhere in it (a flag, null and text are not)."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_holds_numbers(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_numbers(v) for v in value)
    return False


def _keys_with_numbers(result: Any) -> list[str]:
    """Every key, at any depth, whose value holds a number."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if _holds_numbers(v):
                    out.append(str(k))
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(result)
    return out


def named_in_results(name: str, result: Any) -> set[str]:
    """The keys of the run's results (at any depth, holding a number) that are the quantity the plan calls ``name``: the
    same name, a name that begins with it (``metric_a_unit``, ``metric_a_interval``), a name the plan's is a
    longer form of (``metric_b`` for ``metric_b_by_setting``), and what begins with a key found that
    way (the interval of the same quantity). Case, punctuation and underscores are ignored. A key of fewer than six
    characters is never the plan's name written longer."""
    want = _norm(name)
    if len(want) < 2:
        return set()
    keys = {k: _norm(k) for k in _keys_with_numbers(result)}
    base = {k for k, n in keys.items() if n and (n.startswith(want) or (len(n) >= 6 and want.startswith(n)))}
    stems = {keys[k] for k in base if len(keys[k]) >= 4}
    return base | {k for k, n in keys.items() if any(n.startswith(stem) for stem in stems)}


def absent_from_results(names: list[str], result: Any) -> list[str]:
    """Of the plan's quantities ``names``, those the run's results hold no number for under any key that contains the name
    (case, punctuation and underscores ignored). Containment, not equality: an analysis that writes ``variant_metric_c``
    for ``metric_c`` has the quantity, and a stop must never rest on a name written another way."""
    keys = [_norm(k) for k in _keys_with_numbers(result)]
    out = []
    for name in names:
        want = _norm(name)
        if want and not any(want in k or (len(k) >= 6 and k in want) for k in keys):
            out.append(name)
    return out


_FAILURE_KEYS = frozenset({"error", "errors", "failure", "failures", "failed", "exception"})
_FAILURE_WORDS = re.compile(r"invalid|fail|error|incomplete|abort|not[_ ]?run|no[_ ]?data", re.IGNORECASE)


def reports_failure(result: Any) -> str:
    """What the run's own result says about itself when it reports it did not produce its results (a key such as ``errors``
    holding something, or a ``status`` that says invalid, failed or incomplete), else ``""``. The script's word about its
    own failure; the engine does not guess one."""
    found = ""

    def walk(node: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                n = _norm(k)
                if n in _FAILURE_KEYS and v not in (None, False, "", [], {}, 0) and not _holds_numbers(v):  # text, not error statistics
                    found = f"{k}: {str(v)[:120]}"
                    return
                if n == "status" and isinstance(v, str) and _FAILURE_WORDS.search(v):
                    found = f"{k}: {v[:120]}"
                    return
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(result)
    return found


#: The kinds of requirement that name one quantity of the plan (a dependent variable, a metric).
QUANTITY_KINDS = ("dependent variable", "metric")


def differing_quantities(quest_root: Path, code_dir: Path) -> list[dict[str, Any]]:
    """The quantities the plan names that a second model found the final version of the code computes differently (or not
    at all) and that the repairs did not change: ``{"name", "kind", "status", "text", "note"}``, one per name. Only a finding
    of the version that ran (:func:`not_computed`), only one the run's results did not contradict, and only a requirement
    that names a quantity. A problem of logic is not one (its ``name`` lists several things)."""
    seen: dict[str, dict[str, Any]] = {}
    for v in not_computed(quest_root, code_dir):
        name = str(v.get("name") or "").strip()
        if (v.get("kind") not in QUANTITY_KINDS or v.get("status") not in (DIFFERENT, MISSING) or not name
                or v.get("contradicted") or name in seen):
            continue
        seen[name] = {"name": name, "kind": v.get("kind"), "status": v.get("status"), "text": str(v.get("text") or ""),
                      "note": str(v.get("note") or "")}
    return list(seen.values())


def resolve_headline(quest_root: Path, code_dir: Path, result: Any) -> list[str]:
    """After a run: of the headline requirements a reader still called missing, those the run's results do not hold
    either (the quest then stops, honestly). The ones the results do hold are recorded as a disagreement between the
    reader and the results (the reader was wrong, and the paper is not told they were not computed)."""
    record = load(quest_root)
    if not record.get("final") or record.get("version") != version(code_dir):
        return []
    absent: list[str] = []
    changed = False
    for v in record.get("problems") or []:
        if not isinstance(v, dict) or not v.get("headline") or v.get("contradicted") or v.get("kind") == "logic":
            continue
        if in_results(str(v.get("name") or ""), result):
            if v.get("status") == DIFFERENT:
                continue  # computed under another definition: the results holding a value does not say the reader was wrong
            v["contradicted"] = True
            changed = True
        else:
            absent.append(v["text"])
    if changed:
        record["disagreement"] = [v["text"] for v in record["problems"] if v.get("contradicted")]
        save(quest_root, record)
    return absent


def _numbers(value: Any) -> list[float]:
    """Every number in a value (a number, or a list or mapping of them), booleans left out."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [n for x in value for n in _numbers(x)]
    if isinstance(value, dict):
        return [n for x in value.values() for n in _numbers(x)]
    return []


def constant_in(name: str, result: Any) -> bool | None:
    """Whether the run's results hold ``name`` (ignoring case and punctuation) at the same value everywhere it appears:
    ``True`` when it does (at least two values, all equal), ``False`` when they differ, ``None`` when the results cannot
    tell (the name is absent, or it appears once). This is the one claim of a reader about the logic that the results can
    confirm or rule out."""
    want = _norm(name)
    if not want:
        return None
    values: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if _norm(k) == want:
                    values.extend(_numbers(v))
                else:
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for x in node:
                walk(x)

    walk(result)
    if len(values) < 2:
        return None
    return all(abs(x - values[0]) <= 1e-12 * max(1.0, abs(values[0])) for x in values)


def resolve_logic(quest_root: Path, code_dir: Path, result: Any) -> list[str]:
    """After a run: of the problems of logic a reader still names after the repairs, those that say a quantity is a
    constant are decided by the run's results. A quantity the results hold at differing values contradicts the reader (a
    disagreement, recorded, and the paper is not told); one they hold at one value everywhere confirms it. Returns the
    texts of the confirmed ones that are about a headline quantity and were called high severity: only then does the
    quest stop. A finding the results cannot decide (a normalisation, a statistic, a skipped step) never stops the
    quest on the reader's word: the limitations say that a second model flagged it and it was not resolved."""
    record = load(quest_root)
    if not record.get("final") or record.get("version") != version(code_dir):
        return []
    confirmed: list[str] = []
    changed = False
    for v in record.get("problems") or []:
        if not isinstance(v, dict) or v.get("kind") != "logic" or not v.get("constant") or v.get("contradicted"):
            continue
        verdict = constant_in(str(v.get("name") or ""), result)
        if verdict is False:
            v["contradicted"] = True
            changed = True
        elif verdict is True:
            if not v.get("confirmed"):
                v["confirmed"] = True
                changed = True
            if v.get("headline") and v.get("severity") == "high":
                confirmed.append(v["text"])
    if changed:
        record["disagreement"] = [v["text"] for v in record["problems"] if v.get("contradicted")]
        save(quest_root, record)
    return confirmed


def write_note(quest_root: Path, code_dir: Path) -> str:
    """For the write prompt: what the code does not compute, to be said plainly in the limitations. ``""`` when nothing."""
    record = load(quest_root)
    notes: list[str] = []
    if record and not record.get("read_version"):
        # The code that runs was never read by another model (the reading failed, the same model was the reader, or the
        # readings allowed are spent): the writer says so.
        why = record.get("error") or record.get("skipped") or "no reading was made"
        notes.append("The code that produced these results was not compared with the plan by another model "
                     f"({why}): say so plainly in the limitations.")
    elif record.get("read_version") and record["read_version"] != version(code_dir):
        notes.append("The code was read against the plan by a second model, but it was changed afterwards and the final "
                     "version was not read again by another model: say so in the limitations.")
    elif record.get("version") == version(code_dir) and record.get("not_reviewed"):
        names = "; ".join(str(x) for x in record["not_reviewed"][:8])
        notes.append("A second model was asked to compare the code with the plan, and did not check these requirements: "
                     f"{names}. Say plainly (in the limitations) that they were not checked against the code by another model"
                     + (", the study's main quantity among them." if record.get("not_reviewed_headline") else "."))
    still = [v for v in not_computed(quest_root, code_dir) if not v.get("contradicted")]
    flagged = [v for v in still if v.get("kind") == "logic"]
    left = [v for v in still if v.get("kind") != "logic"]
    if flagged:
        listed = "; ".join(f"{v['text']}" + (f" (in {v['file']}" + (f", line {v['line']}" if v.get("line") else "") + ")"
                                             if v.get("file") else "") + f" [{v.get('severity', 'medium')}]"
                           for v in flagged[:8])
        notes.append("A second model read how the code computes what the plan says, and the code was sent back to be "
                     f"repaired; it still flags: {listed}. The run's results cannot settle this, and the reader can be wrong: "
                     "say plainly in the limitations that a second model flagged these and they were not resolved, and do "
                     "not present the quantities they concern as established.")
    if record.get("converged") is False and (flagged or [v for v in still if v.get("kind") != "logic"]):
        notes.append("The repairs of the second model's findings were stopped because a round did not reduce the open "
                     "high-severity findings; what is open is listed here as it stands.")
    if not left:
        return " ".join(notes)
    names = "; ".join(f"{v['text']} ({'not computed' if v['status'] == MISSING else 'computed differently from the plan'})"
                      for v in left[:8])
    return ("A second model compared the code with what the plan requires, and the code was sent back to be repaired; "
            f"these requirements are still not met: {names}. Say so plainly (in the limitations): a quantity listed as "
            "not computed was not computed, and do not write a value, a figure or a claim for it.")


def plain_lines(record: dict[str, Any]) -> str:
    """One sentence for run.log about a record."""
    if record.get("skipped"):
        return f"the code was not read by another model ({record['skipped']})"
    if record.get("error"):
        return f"the code could not be compared with the plan ({record['error']})"
    items = [v for v in record.get("problems") or [] if isinstance(v, dict)]
    n = sum(1 for v in items if v.get("kind") != "logic")
    m = len(items) - n
    return f"the code was read by {record.get('reviewer') or 'another model'}: " + (
        (f"{n} requirement(s) of the plan are missing or implemented differently" if n else "")
        + (" and " if n and m else "")
        + (f"{m} problem(s) in how it computes what the plan says" if m else "")
        if items else "every requirement is met and no problem was found")


