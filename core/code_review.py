"""A different model reads the generated code against the plan's required outputs, before the study runs.

A script can run to the end, print a plausible ``RESULT_JSON`` and still leave out a quantity the plan requires (a
dependent variable, a metric, a planned figure, a swept parameter, the equation a function was meant to compute). The
model that wrote the code reads its own code as the plan says; nothing else checks it. Here, after the code is written
(and the model's functions are checked one at a time) and before the equation tests and the oracle gate, the model
named for ``oracle_review`` (the same routing as the second reading of the plan's checks, so it is the other provider
when one is configured) reads the plan's required outputs against the scripts and says, per requirement, whether it is
**implemented**, **missing** or **implemented differently**, with the file and line.

* A reader that is the model that wrote the code is not a second reader: nothing is asked, and ``.fi/code_review.json``
  and ``run.log`` say that the code was not read by another model.
* What is missing or different goes to the existing repair path (a failed run that ``execute_reflect`` repairs) as a
  plain directive naming the requirement; never an expected value (the requirements are names, never numbers the plan
  wants reproduced). At most :data:`MAX_SEND_BACKS` times in a quest, and at most one review per code version
  (:func:`version`); at most :data:`MAX_REVIEWS` reviews in a quest, so the cost is bounded.
* What is still missing afterwards is not reported: the paper's limitations say plainly that it was not computed
  (:func:`write_note`). A missing *headline* quantity (the first dependent variable, the metric the precision target
  is for) ends in the honest stop (``stuck_no_findings``).
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any

from . import oracle_check as _oracle

RECORD = Path(".fi") / "code_review.json"
#: Reviews of the code in a whole quest (one per code version), and the times a review sends the code back for repair.
MAX_REVIEWS = 4
MAX_SEND_BACKS = 2
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


def requirements(design: dict[str, Any] | None, protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The plan's required outputs, as ``{"id", "kind", "text", "headline"}`` (R1, R2 ...): the dependent variables, the
    metrics, the planned figures, the swept parameters, the checks' measures and the equations the simulation computes
    its data with. Names and descriptions only, never a number the code must reproduce."""
    design = design if isinstance(design, dict) else {}
    protocol = protocol if isinstance(protocol, dict) else {}
    out: list[dict[str, Any]] = []

    def add(kind: str, text: str, headline: bool = False) -> None:
        text = _text(text)
        if text and not any(r["kind"] == kind and r["text"] == text for r in out):
            out.append({"id": f"R{len(out) + 1}", "kind": kind, "text": text, "headline": headline})

    precision = protocol.get("precision") if isinstance(protocol.get("precision"), dict) else {}
    headline_metric = _text(precision.get("metric")).lower()
    variables = design.get("variables") if isinstance(design.get("variables"), dict) else {}
    dependent = [d for d in (variables.get("dependent") if isinstance(variables.get("dependent"), list) else []) if _text(d)]
    for i, name in enumerate(dependent):
        add("dependent variable", f"the dependent variable {_text(name)} is computed and reported in the result", i == 0)
    for m in protocol.get("metrics") if isinstance(protocol.get("metrics"), list) else []:
        if isinstance(m, dict) and _text(m.get("id")):
            mid = _text(m["id"])
            est = _text(m.get("estimand"))
            add("metric", f"the result reports {mid}" + (f" ({est})" if est else ""),
                bool(headline_metric) and (mid.lower() == headline_metric or headline_metric in mid.lower()))
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


def parse(reply: Any, reqs: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
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
                    "status": status, "file": _text(item.get("file"), 80),
                    "line": line if isinstance(line, int) and not isinstance(line, bool) else None,
                    "note": _text(item.get("note"))})
    return out or None


def problems(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [v for v in verdicts if v["status"] in (MISSING, DIFFERENT)]


def script_for(problems: list[dict[str, Any]], code_dir: Path) -> str | None:
    """The script a repair should rewrite for these problems: ``simulate.py`` when the reader pointed there, or all of
    them are about the simulation (a swept parameter, a check's measure, an equation); else the analysis (``None``)."""
    if not (Path(code_dir) / "simulate.py").is_file() or not problems:
        return None
    files = {v["file"].replace("\\", "/").split("/")[-1] for v in problems if v.get("file")}
    if "simulate.py" in files or (not files and all(v["kind"] in ("sweep", "check", "equation") for v in problems)):
        return "simulate.py"
    return None


def directive(items: list[dict[str, Any]]) -> str:
    """What stands where a traceback would in the repair request: plain, naming each requirement, never a number."""
    lines = []
    for v in items:
        what = "is missing from the code" if v["status"] == MISSING else "is implemented differently from the plan"
        where = f" (see {v['file']}" + (f", line {v['line']}" if v["line"] else "") + ")" if v["file"] else ""
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
    return [v["text"] for v in not_computed(quest_root, code_dir) if v.get("headline")]


def write_note(quest_root: Path, code_dir: Path) -> str:
    """For the write prompt: what the code does not compute, to be said plainly in the limitations. ``""`` when nothing."""
    left = not_computed(quest_root, code_dir)
    if not left:
        return ""
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
    n = len(record.get("problems") or [])
    return f"the code was read by {record.get('reviewer') or 'another model'}: " + (
        f"{n} requirement(s) of the plan are missing or implemented differently" if n else "every requirement is met")


