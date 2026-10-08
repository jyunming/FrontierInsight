"""One issue is one thing: the findings of the code reader, kept by a stable id across readings.

The reader's wording changes from one reading to the next, and a requirement that is "different" and a fault of logic
about the same quantity are one fault told twice. Counting findings per reading therefore says little: the same issue
comes back reworded, and a list that is cut short at a fixed length rotates instead of shrinking. Here every finding gets
an id that does NOT depend on the reader's words:

    hash(file, the function that encloses the place it names, the kind of finding, the planned quantity or requirement
    it concerns)

Findings with one id are one issue. Requirement findings (a quantity, a metric) and findings of logic about the same
quantity in the same function merge. The record ``.fi/code_review.json`` keeps, per id, the full history: when it was
first seen, each reading's status, whether a test showed it, the repairs tried (which model, how it ended) and the
outcome (:func:`update_issues`). :func:`ranked` puts the findings that can change a reported number first.

Pure functions over the record and the code on disk; no model is called here and the engine is not imported.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

#: How an issue ended (the ``outcome`` of an issue). ``open`` and ``reopened`` are still to be worked on.
OPEN, REOPENED = "open", "reopened"
FIXED = "fixed"
NOT_DEMONSTRATED = "not demonstrated"
PLAN_PROBLEM = "plan problem"
GAVE_UP = "not repaired"
GONE = "no longer flagged"
#: Outcomes from which nothing more is tried.
FINAL = (FIXED, NOT_DEMONSTRATED, PLAN_PROBLEM, GAVE_UP, GONE)

#: The requirement kinds that name one quantity of the plan; for them the id rests on the quantity's name.
QUANTITY_KINDS = ("dependent variable", "metric")
_SEVERITY = {"high": 0, "medium": 1, "low": 2}
_WORDINGS = 6
#: Where several quantities are written in one finding (``a, b`` / ``a and b`` / ``a/b``).
_SPLIT = re.compile(r"\s*(?:,|;|/|&|\+|\band\b)\s*", re.IGNORECASE)


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


# --- the function a finding is in ---------------------------------------------------------------------------------------


class Code:
    """The files of ``code/`` read once, to ask what is at a place (a file and a line)."""

    def __init__(self, code_dir: Path) -> None:
        from . import code_layout as _layout

        self.dir = Path(code_dir)
        self.sources: dict[str, str] = {}
        for name in ("simulate.py", "experiment.py"):
            path = self.dir / name
            if path.is_file():
                self.sources[name] = path.read_text(encoding="utf-8", errors="replace")
        self.sources.update(_layout.package_sources(self.dir))
        self._trees: dict[str, ast.Module | None] = {}
        self._by_name: dict[str, str] = {}
        for rel in self.sources:
            self._by_name.setdefault(Path(rel).name.lower(), rel)

    def resolve(self, name: str) -> str:
        """The path under ``code/`` of the file called ``name`` (``""`` when the code has none by that name)."""
        name = str(name or "").replace("\\", "/")
        if name in self.sources:
            return name
        return self._by_name.get(Path(name).name.lower(), "") if name else ""

    def tree(self, rel: str) -> ast.Module | None:
        if rel not in self._trees:
            try:
                self._trees[rel] = ast.parse(self.sources[rel])
            except (KeyError, SyntaxError, ValueError):
                self._trees[rel] = None
        return self._trees[rel]

    def function_at(self, rel: str, line: int | None) -> str:
        """The innermost function (``Class.method`` for a method) that holds ``line``; ``<module>`` outside any."""
        tree = self.tree(rel) if rel else None
        if tree is None or not line:
            return ""
        best: tuple[int, str] | None = None

        def walk(node: ast.AST, prefix: str) -> None:
            nonlocal best
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    first = min([child.lineno, *[d.lineno for d in child.decorator_list]])
                    name = f"{prefix}{child.name}"
                    if first <= line <= (child.end_lineno or child.lineno):
                        size = (child.end_lineno or child.lineno) - first
                        if best is None or size <= best[0]:
                            best = (size, name)
                    walk(child, name + ".")
                elif isinstance(child, ast.ClassDef):
                    walk(child, f"{prefix}{child.name}.")
                else:
                    walk(child, prefix)

        walk(tree, "")
        return best[1] if best else "<module>"

    def anchor_at(self, rel: str, line: int | None) -> str:
        """What the statement at ``line`` assigns (or ``return``): a name read from the code, never from the reader's
        words. Used only when a finding names no quantity."""
        tree = self.tree(rel) if rel else None
        if tree is None or not line:
            return ""
        found = ""
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt) or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.lineno <= line <= (node.end_lineno or node.lineno):
                if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
                    found = node.targets[0].id
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    found = node.target.id
                elif isinstance(node, ast.Return):
                    found = "return"
        return found


# --- the id --------------------------------------------------------------------------------------------------------------


def kind_class(v: dict[str, Any]) -> str:
    """The kind of a finding as far as the id is concerned: a requirement about a quantity and a fault of logic about a
    quantity are the same kind (they are one issue told twice); a figure, a swept setting, a check and an equation each
    have their own."""
    kind = str(v.get("kind") or "")
    if kind == "logic":
        return "quantity" if str(v.get("name") or "").strip() else "logic"
    if kind in QUANTITY_KINDS:
        return "quantity"
    return kind or "requirement"


def quantities(v: dict[str, Any]) -> list[str]:
    """The planned quantities or requirements a finding concerns, normalised, in the order written."""
    kind = str(v.get("kind") or "")
    if kind == "logic" or kind in QUANTITY_KINDS:
        parts = [norm(p) for p in _SPLIT.split(str(v.get("name") or "")) if norm(p)]
        return parts
    text = norm(v.get("text"))
    return [text] if text else []


def _id(file: str, function: str, kind: str, quantity: str) -> str:
    raw = "|".join([file.lower(), function, kind, quantity])
    return "F-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def index_problems(problems: list[dict[str, Any]], code_dir: Path) -> None:
    """Give each problem of a reading its stable id (``fid``), in place. A finding of logic that names several
    quantities takes the id one of them already has from a requirement in the same function (the two are one issue),
    else the id of the first."""
    code = Code(code_dir)
    taken: set[str] = set()
    placed: list[tuple[dict[str, Any], str, str, str, list[str]]] = []
    for v in problems:
        rel = code.resolve(str(v.get("file") or ""))
        line = v.get("line") if isinstance(v.get("line"), int) else None
        function = code.function_at(rel, line)
        qs = quantities(v)
        if not qs and str(v.get("kind")) == "logic":
            qs = [code.anchor_at(rel, line)]
        placed.append((v, rel, function, kind_class(v), qs or [""]))
    for v, rel, function, kind, qs in placed:
        if str(v.get("kind")) != "logic":
            v["fid"] = _id(rel, function, kind, qs[0])
            taken.add(v["fid"])
    for v, rel, function, kind, qs in placed:
        if str(v.get("kind")) != "logic":
            continue
        ids = [_id(rel, function, kind, q) for q in qs]
        v["fid"] = next((i for i in ids if i in taken), ids[0])
        v["file_rel"], v["function"], v["quantities"] = rel, function, qs
    for v, rel, function, _kind, qs in placed:
        v.setdefault("file_rel", rel)
        v.setdefault("function", function)
        v.setdefault("quantities", qs)


# --- the history of an issue ---------------------------------------------------------------------------------------------


def _flag(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def changes_result(v: dict[str, Any]) -> bool:
    """Whether the reader says the finding can change a reported number or figure; for a finding that does not say, a
    high severity or a requirement about a quantity does."""
    if v.get("changes_result") is not None:
        return bool(v["changes_result"])
    if str(v.get("kind")) == "logic":
        return v.get("severity") == "high"
    return str(v.get("kind")) in QUANTITY_KINDS


def severity_of(v: dict[str, Any]) -> str:
    """How serious a problem is: a finding of logic says so itself; a requirement that is missing or different is high
    when it is about a headline quantity, else medium (as :func:`core.code_review.severity_of`)."""
    if v.get("kind") == "logic":
        return v.get("severity") if v.get("severity") in _SEVERITY else "medium"
    return "high" if v.get("headline") else "medium"


def issues_of(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    issues = record.get("issues")
    return issues if isinstance(issues, dict) else {}


def update_issues(record: dict[str, Any], problems: list[dict[str, Any]], n: int, version: str = "") -> None:
    """Fold one reading's problems (each with its ``fid``) into ``record["issues"]``: a new id is a new issue, a known id
    is the same issue seen again, whatever it was called; an issue the reading does not flag is marked so. An issue that
    had been fixed and is flagged again is reopened (once more tried, within its own bound)."""
    issues = record.setdefault("issues", {})
    seen: set[str] = set()
    by_id: dict[str, list[dict[str, Any]]] = {}
    for v in problems:
        if v.get("fid"):
            by_id.setdefault(v["fid"], []).append(v)
    for fid, group in by_id.items():
        seen.add(fid)
        first = group[0]
        issue = issues.get(fid)
        if issue is None:
            issue = issues[fid] = {
                "id": fid, "first_seen": n, "readings": [], "demonstrated": None, "outcome": OPEN, "repairs": [],
                "tests": 0, "wordings": [], "fixed_by": "", "calls": {"test": 0, "repair": 0, "other": 0}, "rounds": 0}
        issue["last_seen"] = n
        kinds = sorted({str(g.get("kind")) for g in group})
        issue.update({
            "kind": kind_class(first), "kinds": kinds, "file": first.get("file_rel") or first.get("file") or "",
            "function": first.get("function") or "", "quantity": next((q for g in group for q in g.get("quantities") or [] if q), ""),
            "severity": min((severity_of(g) for g in group), key=lambda s: _SEVERITY.get(s, 1)),
            "changes_result": any(changes_result(g) for g in group),
            "headline": any(bool(g.get("headline")) for g in group),
            "plan_conflict": any(bool(g.get("plan_conflict")) for g in group),
            "where": next((g.get("where") for g in group if g.get("where")), ""),
            "text": first.get("text", ""), "why": next((g.get("note") for g in group if g.get("note")), ""),
            "fix": next((g.get("fix") for g in group if g.get("fix")), ""),
            "problem_ids": [g.get("id") for g in group],
            "statement": _statement(group),
        })
        for g in group:
            text = " ".join(str(g.get("text") or "").split())[:300]
            if text and text not in issue["wordings"]:
                issue["wordings"] = [*issue["wordings"], text][-_WORDINGS:]
        issue["readings"].append({"n": n, "status": "open", "version": version})
        if issue["outcome"] in (FIXED, GONE):
            issue["outcome"] = REOPENED
            issue["rounds"] = int(issue.get("rounds") or 0) + 1
            issue["demonstrated"] = None
    for fid, issue in issues.items():
        if fid in seen:
            continue
        issue["readings"].append({"n": n, "status": "not flagged", "version": version})
        if issue["outcome"] in (OPEN, REOPENED):
            issue["outcome"] = GONE


def _statement(group: list[dict[str, Any]]) -> str:
    """The problems of one issue as the requests show them, each in full (where, what, why, how to fix)."""
    from . import code_review as _review

    return "\n\n".join(re.sub(r"^1\. ", "", _review.findings_text([g])) for g in group)


def _open_now(issue: dict[str, Any]) -> bool:
    readings = issue.get("readings") or []
    return bool(readings) and readings[-1].get("status") == "open"


def open_issues(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The issues the latest reading flags."""
    return [i for i in issues_of(record).values() if isinstance(i, dict) and _open_now(i)]


def rank_key(issue: dict[str, Any]) -> tuple[Any, ...]:
    """Most to least urgent: what can change a reported number first, then severity, then the headline quantity, then
    the one seen first."""
    return (not issue.get("changes_result"), _SEVERITY.get(str(issue.get("severity")), 1), not issue.get("headline"),
            int(issue.get("first_seen") or 0), str(issue.get("id")))


def ranked(record: dict[str, Any], *, only_open: bool = True) -> list[dict[str, Any]]:
    pool = open_issues(record) if only_open else [i for i in issues_of(record).values() if isinstance(i, dict)]
    return sorted(pool, key=rank_key)


def changing_open(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Open issues that can change a result and are not shown to be harmless: the ones the repairs are for. An issue
    whose test did not fail on the current code (not demonstrated) and a contradiction of the plan itself are left out."""
    return sorted((i for i in open_issues(record) if i.get("changes_result") and i.get("demonstrated") is not False
                   and not i.get("plan_conflict") and i.get("outcome") not in (FIXED, GONE)), key=rank_key)


def summary(record: dict[str, Any]) -> dict[str, Any]:
    """Counts for the reports: distinct issues, those that change a result, how many a test demonstrated, how many were
    repaired (and by which models), and what is still open."""
    every = [i for i in issues_of(record).values() if isinstance(i, dict)]
    opened = open_issues(record)
    changing = [i for i in every if i.get("changes_result")]
    demonstrated = [i for i in changing if i.get("demonstrated") is True or i.get("outcome") == FIXED]
    models: dict[str, int] = {}
    for i in every:
        if i.get("outcome") == FIXED and i.get("fixed_by"):
            models[i["fixed_by"]] = models.get(i["fixed_by"], 0) + 1
    return {
        "issues": len(every), "result_changing": len(changing), "demonstrated": len(demonstrated),
        "not_demonstrated": sum(1 for i in every if i.get("outcome") == NOT_DEMONSTRATED),
        "fixed": sum(1 for i in every if i.get("outcome") == FIXED), "fixed_by": models,
        "plan_problems": sum(1 for i in every if i.get("outcome") == PLAN_PROBLEM or i.get("plan_conflict")),
        "open": len(opened),
        "open_result_changing_demonstrated": sum(
            1 for i in opened if i.get("changes_result") and i.get("demonstrated") is True
            and i.get("outcome") not in (FIXED, GONE)),
    }


def convergence_line(record: dict[str, Any]) -> str:
    """One sentence on whether the repairs converged, from the issues a test demonstrated that change a result (the count of
    findings per reading is not used: the same issue comes back reworded)."""
    s = summary(record)
    if not s["issues"]:
        return ""
    return (f"{s['issues']} distinct issue(s), {s['result_changing']} can change a reported result; {s['demonstrated']} of "
            f"those were shown by a failing test and {s['fixed']} issue(s) were repaired; "
            f"{s['open_result_changing_demonstrated']} shown result-changing issue(s) are still open")


#: The parts of the plan's protocol in which a contradiction about a definition can sit, and so the only parts the plan's
#: model may rewrite to resolve one. A check, a threshold, a tolerance and a precision target are never among them.
PLAN_PARTS = ("metrics", "ci_method", "model")


def plan_conflict_request(topic: Any, conflicts: list[tuple[str, str]], values: list[float] | None = None) -> str:
    """The request to the plan's model to make the plan consistent: each contradiction as the reader found it (what and
    why, with the plan's own check values taken out), the topic as the specification, and what it may change."""
    from . import review_pairs as _pairs

    lines = []
    for n, (what, why) in enumerate(conflicts, 1):
        lines.append(f"{n}. {_pairs.scrub(what, values)}" + (f" ({_pairs.scrub(why, values)})" if why else ""))
    return (
        "A second model read the study's code against this plan and found that the plan contradicts itself: two parts of "
        "it define the same thing differently, so no code can satisfy both.\n\n"
        + "\n".join(lines)
        + "\n\nThe topic is the specification:\n\n" + str(topic or "")[:1500]
        + "\n\nRewrite the plan so that each of these is defined in ONE way, the way the topic asks for, and say that "
        "definition in every place the plan mentions it. Change only `protocol.metrics`, `protocol.ci_method` and "
        "`protocol.model` (and the prose of the plan that repeats them). Do not change a check, a threshold, a "
        "tolerance, an expected value, a precision target, the grid or the number of runs. Keep everything else the "
        "plan already has.")
