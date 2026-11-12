"""The functions of the study that already ran and passed, kept when the code step is entered again.

When the code is written again after it already ran (the run produced no result and the design is tried once more, the same
design's code is written again after a script that could not run to the end, or a measured finding names a number the run
computed), the model used to be asked for every function of the study again: the simulation's ``trial`` and each ``derive_*``,
``reference_*`` and ``figure_*`` of ``derive.py``. A figure that was drawn and placed in the paper was then drawn again by new
code, and could fail where the first one had not. FI keeps its own record of each function that ran and passed, the way it
keeps the model's equation functions (:mod:`core.function_steps`): the code that ran, what it was written from, and what it
produced. On a new pass a function is kept unless the reason for the pass names it (the design changed something it reads, or a
finding or a failed run names it); the model is asked only for the rest.

The record is ``.fi/kept_bodies.json``. It is written from FI's own records of the last run (``data/results/derived.json``
says, for each function of ``derive.py``, whether it ran and what it made), never from the model's account of it, and only for
code that has not been changed since that run.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import analysis_frame as _analysis
from . import contract as _contract

RECORD = Path(".fi") / "kept_bodies.json"
SCHEMA = 1
#: The name the simulation has in the record: its whole ``simulate.py`` is what ran.
TRIAL = "trial"


def _sha(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load(root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and data.get("schema") == SCHEMA and isinstance(data.get("bodies"), dict) else {}


def _save(root: Path, record: dict[str, Any]) -> None:
    path = Path(root) / RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass


# ---- what a function reads ----------------------------------------------------------------------------------------------------
def reads(design: dict[str, Any], name: str) -> str:
    """A digest of the parts of the plan the function ``name`` of ``derive.py`` reads: when it differs from the one recorded when
    the function ran, the design changed something the function reads. A figure reads its own declaration, the results it shows
    (role, kind, how each is worked out, its reference and true value) and the settings it puts on an axis, in a series or in a
    panel, with their values. A quantity's function reads its declaration, its inputs, and every setting (it returns a value per
    setting)."""
    plan = _analysis.plan_of(design)
    c = _contract.from_design(design, fi_analysis=True)
    axes = {str(k): list(v) for k, v in {**c.settings.compared, **c.settings.in_each_run}.items()}
    kind, _, ident = name.partition("_")

    def result(i: str) -> dict[str, Any]:
        return {"id": i, "role": plan["roles"].get(i), "kind": plan["kinds"].get(i),
                "derived": next((d for d in plan["derived"] if d["id"] == i), None),
                "reference": plan["references"].get(i), "known": plan["known"].get(i)}

    if kind == "figure" and ident in plan["figures"]:
        fig = plan["figures"][ident]
        named = {str(x) for x in (fig.get("x"), fig.get("series"), *(fig.get("settings") or {})) if x}
        return _sha({"figure": fig, "shows": [result(i) for i in fig["shows"]],
                     "axes": {k: v for k, v in axes.items() if k in named}, "compared": plan["compared"]})
    d = next((d for d in plan["derived"] if d["by"] == name), None)
    if d is None:
        return ""
    return _sha({"derived": d, "inputs": [result(i) for i in d["inputs"]], "axes": axes, "parts": plan["parts"],
                 "held": plan["held"], "fixed": plan["fixed"]})


# ---- the study's part of derive.py ----------------------------------------------------------------------------------------------
def _spans(text: str) -> dict[str, tuple[int, int]]:
    """``{function: (first line, last line)}`` (1-based, decorators included) of the study's functions of ``derive.py``."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith(_analysis._BODY_PREFIXES):
            out[n.name] = (min([n.lineno, *(d.lineno for d in n.decorator_list)]), n.end_lineno or n.lineno)
    return out


def sources(text: str) -> tuple[dict[str, str], str]:
    """``({function: its source}, a digest of everything else the study wrote)``: the digest covers the helpers and imports the
    functions use, outside FI's frame and outside the functions themselves."""
    text = (text or "").lstrip("﻿")
    lines = text.splitlines()
    spans = _spans(text)
    out = {n: "\n".join(lines[a - 1:b]) + "\n" for n, (a, b) in spans.items()}
    frame = _analysis._frame_span(text)
    drop = {i for a, b in spans.values() for i in range(a - 1, b)}
    if frame is not None:
        drop |= set(range(text[:frame[0]].count("\n"), text[:frame[1]].count("\n")))
    rest = "\n".join(ln.rstrip() for i, ln in enumerate(lines) if i not in drop and ln.strip())
    return out, _sha(rest)


def put_back(text: str, name: str, source: str) -> str:
    """``text`` (a ``derive.py``) with the function ``name`` as ``source``: replaced where it is, else added at the end."""
    spans = _spans(text)
    lines = text.splitlines()
    new = source.rstrip("\n").splitlines()
    if name in spans:
        a, b = spans[name]
        lines[a - 1:b] = new
        return "\n".join(lines) + "\n"
    return text.rstrip("\n") + "\n\n\n" + "\n".join(new) + "\n"


# ---- the record -----------------------------------------------------------------------------------------------------------------
#: FI's receipt of the last analysis run (written by :func:`write_receipt` from the run's own copy): the files the analysis and
#: the simulation had when it started, the trial data it was given, and the record it wrote. It replaces the time a file was
#: last written, which a checkout, a copy or a clock can move without the content changing (or the other way round).
RECEIPT = Path(".fi") / "analysis_ran.json"


def receipt_inputs(root: Path, summary: str | None) -> dict[str, Any]:
    """The inputs of an analysis run as they are now (core/reuse_key.py): its part of the quest, the simulation part and the
    trial summary it is given (``summary``, relative to the quest)."""
    from . import reuse_key as _reuse_key

    sim = _reuse_key.files(root, "simulation", data=False)
    ana = _reuse_key.files(root, "analysis")
    trials = None
    if summary:
        # every file of the folder the analysis is given as FI_RAW_DIR (the summary and the ledger beside it), not the summary alone
        try:
            folder = (Path(root) / summary).parent
            found = {p.relative_to(root).as_posix(): _reuse_key.digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}
            trials = _reuse_key.digest_text(json.dumps(found, sort_keys=True)) if (Path(root) / summary).is_file() else None
        except OSError:
            trials = None
    return {"simulation": sim.digests, "analysis": ana.digests, "trials": trials, "summary": summary,
            "complete": sim.complete and ana.complete and (not summary or trials is not None)}


def write_receipt(root: Path, before: dict[str, Any], dropped: list[str] | tuple[str, ...] = ()) -> bool:
    """After an analysis run that returned 0: the receipt, only when the files it started with are still what they are (the
    run changed none of its own code) and it wrote its record. Otherwise no receipt (the earlier one is removed), so no
    function of that run is kept: it is written again, never a stop. ``dropped``: the paths the run wrote into FI's own folders
    that the copy did not bring back (core/outside_run.py ``left_out``): a write under ``code/`` means the analysis changed its own
    code while it ran, which the quest's files cannot show."""
    return record_run(root, before, dropped) is None


def record_run(root: Path, before: dict[str, Any], dropped: list[str] | tuple[str, ...] = ()) -> str | None:
    """:func:`write_receipt`, saying why no receipt was written: ``None`` when it was; ``""`` when there is nothing to keep (the
    analysis is not FI's: there is no ``derive.py`` and it wrote no record of FI's), so nothing needs saying; else the reason in
    plain words, for run.log."""
    from . import reuse_key as _reuse_key

    root = Path(root)
    (root / RECEIPT).unlink(missing_ok=True)
    derived = root / _analysis.DERIVED_PATH
    if not derived.is_file():
        if not (root / "code" / _analysis.DERIVE_FILE).is_file():
            return ""
        return f"the analysis wrote no record of its run ({_analysis.DERIVED_PATH.as_posix()})"
    if not before.get("complete"):
        return "a file the analysis started with could not be read"
    if any(str(d).startswith("code/") for d in dropped):
        return "the analysis changed its own code while it ran"
    if receipt_inputs(root, before.get("summary")) != before:
        return "the files the analysis started with changed while it ran"
    try:
        (root / RECEIPT).write_text(json.dumps({"schema": 1, "inputs": before, "derived_sha": _reuse_key.digest(derived)}),
                                    encoding="utf-8")
    except OSError as exc:
        return f"FI could not write its receipt of the run ({exc})"
    return None


def _receipt(root: Path) -> tuple[bool, bool]:
    """``(the analysis and its trial data are as the last run had them, the simulation is)``: what a kept function and a kept
    simulation each need. ``(False, False)`` with no receipt."""
    from . import reuse_key as _reuse_key

    try:
        saved = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
        if saved.get("schema") != 1 or saved["derived_sha"] != _reuse_key.digest(root / _analysis.DERIVED_PATH):
            return False, False
        was = saved["inputs"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False, False
    now = receipt_inputs(root, was.get("summary"))
    if not now["complete"] or not was.get("complete"):
        return False, False
    return (now["analysis"] == was.get("analysis") and now["trials"] == was.get("trials"),
            now["simulation"] == was.get("simulation"))


def snapshot(root: Path, design: dict[str, Any]) -> list[str]:
    """Record, from FI's records of the last run, every function that ran and passed in code that has not changed since. Returns
    the names recorded now. What a function was written for (``reads``) is what :func:`mark_written` noted when the code step
    wrote it, never read from the plan as it is now: the plan may have changed since the code ran. An entry for a function the
    plan no longer asks for is dropped."""
    root = Path(root)
    code = root / "code"
    derived_path = root / _analysis.DERIVED_PATH
    record = load(root)
    bodies: dict[str, Any] = dict(record.get("bodies") or {})
    written_for: dict[str, str] = dict(record.get("written_for") or {})
    run = _analysis.read_record(root)
    derive_path, simulate_path = code / _analysis.DERIVE_FILE, code / "simulate.py"
    try:
        derive_text = derive_path.read_text(encoding="utf-8")
        simulate_text = simulate_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        wanted = set(_analysis.body_names(_analysis.plan_of(design)))
    except Exception:  # noqa: BLE001 -- a plan the analysis cannot read keeps nothing
        return []
    have, helpers = sources(derive_text)
    for name in list(bodies):
        if name != TRIAL and (name not in wanted or name not in have):
            bodies.pop(name)
    analysis_as_run, simulation_as_run = _receipt(root)
    fresh = (run.get("schema") == _analysis.DERIVED_SCHEMA and not run.get("from_earlier_run") and not _analysis.stopping_faults(run)
             and analysis_as_run)
    noted: list[str] = []
    if fresh:
        ran = run.get("bodies") if isinstance(run.get("bodies"), dict) else {}
        for name in wanted:
            r = ran.get(name)
            if isinstance(r, dict) and r.get("status") == "ok" and name in have and written_for.get(name):
                bodies[name] = {"kind": name.split("_", 1)[0], "source": have[name], "helpers": helpers,
                                "reads": written_for[name], "produced": r.get("outputs") or [],
                                "seconds": r.get("seconds")}
                noted.append(name)
        # the simulation: the analysis read the record of its trials, so they ran; the file is the one that ran
        if written_for.get(TRIAL) and simulation_as_run and (root / ".fi" / "run_facts.json").is_file():
            try:
                trials = (json.loads((root / ".fi" / "run_facts.json").read_text(encoding="utf-8")).get("trials_run") or {}).get("trials")
            except (OSError, ValueError):
                trials = None
            if trials:
                bodies[TRIAL] = {"kind": TRIAL, "source": simulate_text, "reads": written_for[TRIAL], "produced": [f"{trials} trials"]}
                noted.append(TRIAL)
    if bodies or record:
        _save(root, {"schema": SCHEMA, "bodies": bodies, "written_for": written_for})
    return noted


def mark_written(root: Path, design: dict[str, Any], trial_reads: str, asked: list[str], trial_asked: bool) -> None:
    """Note, when the code step is about to write functions, which part of the plan each was written for: the functions in
    ``asked`` (and the simulation, when ``trial_asked``). The functions kept keep the note they have."""
    record = load(root)
    written_for: dict[str, str] = dict(record.get("written_for") or {})
    for name in asked:
        try:
            written_for[name] = reads(design, name)
        except Exception:  # noqa: BLE001 -- a function whose reads cannot be told is never kept
            written_for.pop(name, None)
    if trial_asked:
        if trial_reads:
            written_for[TRIAL] = trial_reads
        else:
            written_for.pop(TRIAL, None)
    _save(root, {"schema": SCHEMA, "bodies": record.get("bodies") or {}, "written_for": written_for})


@dataclass
class Kept:
    """What a pass keeps: ``names`` (functions of ``derive.py``), ``trial`` (the simulation), and why each of the others is
    asked again."""
    names: list[str] = field(default_factory=list)
    trial: bool = False
    again: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    simulate: str = ""


def decide(root: Path, design: dict[str, Any], asked: list[str], trial_reads: str, named: set[str]) -> Kept:
    """Which of the functions the model would be asked for (``asked``, and the simulation) stay as they ran. ``named``: what the
    reason for this pass names (a function, or ``trial`` for something the simulation computes)."""
    record = load(root)
    out = Kept()
    if not record:
        return out
    try:
        derive_text = (Path(root) / "code" / _analysis.DERIVE_FILE).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        derive_text = ""
    _have, helpers = sources(derive_text)
    for name in [TRIAL, *asked]:
        entry = record["bodies"].get(name)
        if not isinstance(entry, dict):
            continue
        if name in named:
            out.again[name] = "the reason for this pass names it"
        elif name == TRIAL:
            if not trial_reads or entry.get("reads") != trial_reads:
                out.again[name] = "the design changed something the simulation reads"
            else:
                out.trial, out.simulate = True, str(entry.get("source") or "")
        elif entry.get("reads") != reads(design, name):
            out.again[name] = "the design changed something it reads"
        elif entry.get("helpers") != helpers:
            out.again[name] = "the helpers it uses were changed after it ran"
        else:
            out.names.append(name)
            out.sources[name] = str(entry.get("source") or "")
    return out


def restore(root: Path, kept: Kept) -> list[str]:
    """Put the kept functions back as they ran (the file on disk may have been changed by a repair that came after): ``derive.py``
    and ``simulate.py`` are written only when they differ. Returns the files written."""
    code = Path(root) / "code"
    written = []
    if kept.names:
        path = code / _analysis.DERIVE_FILE
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        new = text
        have, _ = sources(text)
        for name in kept.names:
            if have.get(name) != kept.sources[name]:
                new = put_back(new, name, kept.sources[name])
        if new != text:
            path.write_text(new, encoding="utf-8")
            written.append(path.name)
    if kept.trial and kept.simulate:
        path = code / "simulate.py"
        try:
            same = path.read_text(encoding="utf-8") == kept.simulate
        except (OSError, UnicodeDecodeError):
            same = False
        if not same:
            path.write_text(kept.simulate, encoding="utf-8")
            written.append(path.name)
    return written


def say(kept: Kept) -> str:
    """One plain sentence for the run log."""
    parts = []
    names = ([TRIAL] if kept.trial else []) + kept.names
    if names:
        parts.append("kept exactly as they ran and passed: " + ", ".join(names))
    if kept.again:
        parts.append("written again: " + "; ".join(f"{n} ({why})" for n, why in kept.again.items()))
    return "; ".join(parts)
