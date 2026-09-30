"""Make the simulation better against its checks of correctness, one change at a time, and keep a change only when no
check got worse (the improve loop and the ratchet).

After the first full run, FI has measured each of the protocol's criteria itself (:mod:`core.criteria`). When one is
not met and ``engine.improve_rounds`` is above 0, each round goes:

1. The model is shown the criteria, where they stand, the changes already tried and what came of them, and the
   simulation's code; never the study's own results (``RESULT_JSON``, the analysis, the protocol's metrics). It answers
   with ONE edit: one piece of text, found exactly once in one file of the simulation, and what replaces it.
2. The edit is refused before anything runs when it touches a file that is not the simulation's (the analysis script,
   anything outside ``code/``), is not found once, does not parse, changes nothing the simulation computes (comments,
   layout, printed or logged text), repeats a version already tried, writes into the code a value that a check
   expects or a criterion's target, changes how a number a check reads is worked out where the simulation returns it,
   or adds a comparison with a setting a check runs on (:func:`check_edit`).
3. FI runs what the criteria need and computes each one itself: the known-answer cases alone when every criterion is a
   check against a known answer, and all of the protocol's trials when one is about the trials. The files FI judges by
   (the frozen protocol, plan.md, the record of the criteria, the record of the checks, FI's own code that computes
   them), everything in ``code/`` and the loop's own saved copies are hashed before and after; a round that changed
   any of them is aborted, the quest's records and the version kept so far are put back, and the loop stops
   (:func:`guard_hashes`). FI's own files cannot be put back by FI: the log says so.
4. The ratchet (:func:`compare`, :func:`verdict`): each criterion is compared with the version kept so far, on its own
   and by its own ``tolerance``, never through a total. A version is kept as the best only when at least one criterion
   got better by more than its tolerance and none got worse by more than its tolerance. One that got worse is flagged;
   under ``rigor_profile: research`` the quest then stops for a person, otherwise it is only a warning. Either way the
   version kept so far stays.

The loop stops when every criterion is met, when a round makes no criterion better by more than its own tolerance
(nothing left to gain), or when the rounds are used up; the log says which. Each round that ran is a commit in
``code/`` with an entry in ``code/CHANGELOG.md`` naming the round and the criterion values. The version kept is then
run once more the ordinary way, with every check and the full set of trials, and that run's numbers are the ones the
paper reports; whether the study's own results changed is written in the CHANGELOG, as "changed", and is never what
chose the version.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECORD = Path(".fi") / "improve.json"
SNAPSHOTS = Path(".fi") / "improve"
#: Scripts in ``code/`` that are not the simulation: the analysis (it does not touch a criterion), FI's own runner and
#: the scripts of other steps. An edit to one is refused.
NOT_EDITABLE = frozenset({"experiment.py", "run.py", "submit.py", "replot_layout.py", "replot_figures.py",
                          "web_plots.py", "fi_search.py"})
#: FI's own code that computes the criteria: a round's run that changed one of these files is aborted.
_ENGINE_FILES = ("criteria.py", "oracle_check.py", "trial_runner.py", "improve.py")
#: The quest's records a round's run must not change.
_QUEST_FILES = (Path("needs") / "FROZEN_PROTOCOL.json", Path(".fi") / "criteria_history.jsonl",
                Path("needs") / "ORACLE_CHECK.json", Path("plan.md"))  # plan.md: the protocol before it is frozen
_CODE_CAP = 24000

STOP_ALL_MET = "every criterion is met"
STOP_PLATEAU = "the last round made no criterion better by more than its own tolerance"
STOP_BUDGET = "the rounds were used up"
STOP_TAMPERED = "a round changed the files FI judges the code by, so the loop was stopped"
STOP_REGRESSION = "a change made a check of correctness worse and this quest is set up for research"
STOP_NO_MODEL = "the model could not be asked for a change"
STOP_CUT_SHORT = "the loop was cut short before it finished"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt(value: Any) -> str:
    v = _num(value)
    return "not measured" if v is None else f"{v:.6g}"


# --- the simulation's files ------------------------------------------------------------------------------------------


def _package_files(code: Path) -> list[str]:
    """``<package>/<module>.py`` for each module of the model's package in ``code/`` (core/code_layout.py: a folder with
    an ``__init__.py``, the tests aside): the equations the simulation computes with are part of it."""
    out: list[str] = []
    if not code.is_dir():
        return out
    for folder in sorted(code.iterdir()):
        if (folder.is_dir() and not _is_link(folder) and folder.name not in ("tests", "__pycache__", ".git")
                and (folder / "__init__.py").is_file()):
            out += [f"{folder.name}/{f.name}" for f in sorted(folder.glob("*.py")) if f.is_file() and not _is_link(f)]
    return out


def editable(quest_root: Path) -> list[str]:
    """The files of ``code/`` an edit may change: the simulation (``simulate.py``), the helper modules beside it and the
    modules of the model's package (``<package>/model.py``), by their path in ``code/``."""
    code = Path(quest_root) / "code"
    names = sorted(p.name for p in code.glob("*.py") if p.is_file() and p.name not in NOT_EDITABLE) if code.is_dir() else []
    names += _package_files(code)
    return sorted(names, key=lambda n: (n != "simulate.py", "/" not in n, n))


def snapshot(quest_root: Path) -> dict[str, str]:
    """Every ``.py`` file directly in ``code/``, and the modules of the model's package, by their path in ``code/``."""
    code = Path(quest_root) / "code"
    out: dict[str, str] = {}
    paths = [*(sorted(code.glob("*.py")) if code.is_dir() else []), *(code / rel for rel in _package_files(code))]
    for p in paths:
        try:
            # As it is on disk, byte for byte: line endings untouched, and bytes that are not UTF-8 (a file saved in
            # another encoding) carried through unchanged rather than dropped.
            out[p.relative_to(code).as_posix()] = p.read_bytes().decode("utf-8", "surrogateescape")
        except OSError:
            continue
    return out


def restore(quest_root: Path, files: dict[str, str]) -> None:
    """``code/`` back to ``files``: each written as it was, and a ``.py`` file that was not there removed (directly in
    ``code/``, and in the model's package when ``files`` holds the package: a copy saved before the package was part of
    it leaves the package alone). Never written through a link: a link left in place of a file or of the package's
    folder is removed first, and a ``code/`` that is itself a link is not written at all."""
    code = Path(quest_root) / "code"
    if _is_link(code):
        return
    with_package = any("/" in k for k in files)
    present: list[str] = []
    if code.is_dir():
        present = [p.name for p in code.glob("*.py")] + (_package_files(code) if with_package else [])
    for rel in present:
        if rel not in files:
            (code / rel).unlink(missing_ok=True)
    for name, text in files.items():
        path = code / name
        data = text.encode("utf-8", "surrogateescape")
        if path.parent != code and _is_link(path.parent):
            _remove_link(path.parent)  # the package's folder replaced by a link: never written through
        if _is_link(path):
            _remove_link(path)  # a link the run left in place of a file: never written through
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not path.is_file() or path.read_bytes() != data:
                path.write_bytes(data)
        except OSError:
            path.write_bytes(data)
    forget_bytecode(quest_root)


def forget_bytecode(quest_root: Path) -> None:
    """Python's compiled copies of ``code/``: two versions of the same size written in the same second look alike to
    Python's cache (it keys on size and whole-second time), so a run could otherwise import the version before."""
    import shutil

    code = Path(quest_root) / "code"
    if not code.is_dir() or _is_link(code):
        return
    for base, dirs, _names in os.walk(code, followlinks=False):
        if "__pycache__" in dirs:
            cache = Path(base) / "__pycache__"
            if not _is_link(cache):
                shutil.rmtree(cache, ignore_errors=True)
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git") and not _is_link(Path(base) / d)]


def save_snapshot(quest_root: Path, name: str, files: dict[str, str]) -> None:
    """Keep ``files`` as the copy ``name``. What the copy held before is removed without following a link: a link (or a
    junction) inside it is removed itself, never what it points at."""
    import shutil

    folder = Path(quest_root) / SNAPSHOTS / name
    if _is_link(folder):
        _remove_link(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in list(folder.iterdir()):
        if _is_link(old):
            _remove_link(old)
        elif old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
        elif old.suffix == ".py":
            old.unlink(missing_ok=True)
    for file, text in files.items():
        (folder / file).parent.mkdir(parents=True, exist_ok=True)
        (folder / file).write_bytes(text.encode("utf-8", "surrogateescape"))


def load_snapshot(quest_root: Path, name: str) -> dict[str, str] | None:
    folder = Path(quest_root) / SNAPSHOTS / name
    if not folder.is_dir() or _is_link(folder):
        return None
    files, _links = _walk(folder)  # never into a link
    return {p.relative_to(folder).as_posix(): p.read_bytes().decode("utf-8", "surrogateescape")
            for p in sorted(files) if p.suffix == ".py"}


# --- the edit --------------------------------------------------------------------------------------------------------


@dataclass
class Edit:
    file: str
    find: str
    replace: str
    why: str


def parse_edit(obj: Any) -> tuple[Edit | None, str]:
    """The model's answer as an :class:`Edit`, or ``(None, why)``."""
    if not isinstance(obj, dict):
        return None, "the answer was not the JSON object asked for"
    file, find, replace = obj.get("file"), obj.get("find"), obj.get("replace")
    if not isinstance(file, str) or not file.strip():
        return None, "the answer names no file"
    if not isinstance(find, str) or not find:
        return None, "the answer gives no text to find"
    if not isinstance(replace, str):
        return None, "the answer gives no replacement text"
    why = " ".join(str(obj.get("why") or "").split())[:300] or "(no reason given)"
    return Edit(file=file.strip().replace("\\", "/"), find=find, replace=replace, why=why), ""


class _Quiet(ast.NodeTransformer):
    """Takes out what does not change what the code computes: a statement that only prints or logs, and docstrings."""

    _PRINTERS = {"print", "pprint"}
    _LOGGERS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}

    def _prints(self, node: ast.AST) -> bool:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            return False
        func = node.value.func
        if isinstance(func, ast.Name):
            return func.id in self._PRINTERS
        if isinstance(func, ast.Attribute):
            owner = func.value
            if func.attr == "write" and isinstance(owner, ast.Attribute) and owner.attr in ("stdout", "stderr"):
                return True  # sys.stdout.write(...)
            if func.attr in self._LOGGERS and isinstance(owner, ast.Name) and owner.id.lower() in (
                    "logging", "log", "logger", "_log", "_logger", "warnings"):
                return True
            if func.attr == "warn" and isinstance(owner, ast.Name) and owner.id == "warnings":
                return True
        return False

    def _body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        kept = [s for s in body if not self._prints(s)]
        if kept and isinstance(kept[0], ast.Expr) and isinstance(getattr(kept[0], "value", None), ast.Constant) \
                and isinstance(kept[0].value.value, str):
            kept = kept[1:]  # a docstring
        return kept or [ast.Pass()]

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            value = getattr(node, field, None)
            if isinstance(value, list) and value and all(isinstance(s, ast.stmt) for s in value):
                setattr(node, field, self._body(value))
        return node


def normalized(text: str) -> str | None:
    """What the code computes, with comments, layout, printed and logged text and docstrings taken out; ``None`` when it
    does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    tree = _Quiet().visit(tree)
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def _numbers(text: str) -> list[float]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            v = _num(node.value)
            if v is not None:
                out.append(v)
    return out


def _trivial(value: float) -> bool:
    return (value == int(value) and abs(value) <= 10) or value in (0.5, 0.25, 0.1, 0.01, 1e-3)


def planted(old: str, new: str, expected: list[float]) -> list[float]:
    """The values a check expects that the edit writes into the code as a number of its own (one the old code did not
    have): the way to pass a check without being right. Small round numbers (0, 1, 0.5, ...) are not counted."""
    before = _numbers(old)
    found: list[float] = []
    for v in _numbers(new):
        if any(math.isclose(v, b, rel_tol=1e-12, abs_tol=0.0) for b in before):
            continue
        for e in expected:
            if not _trivial(e) and math.isclose(v, e, rel_tol=1e-9, abs_tol=1e-15) and e not in found:
                found.append(e)
    return found


def _measure_exprs(text: str, keys: set[str]) -> list[str]:
    """How the code computes each number a check reads (``{"error": <expression>}``, ``out["error"] = ...``,
    ``dict(error=...)``), as syntax, sorted."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value in keys:
                    out.append(f"{k.value}={ast.dump(v)}")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value in keys:
                    out.append(f"{t.slice.value}={ast.dump(node.value) if node.value is not None else ''}")
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in keys:
                    out.append(f"{kw.arg}={ast.dump(kw.value)}")
    return sorted(out)


def _compares(text: str) -> list[ast.Compare]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    return [n for n in ast.walk(tree) if isinstance(n, ast.Compare)]


def _reads_setting(node: ast.AST, names: set[str], aliases: dict[str, str] | None = None) -> set[str]:
    """The settings a piece of syntax reads: ``cell["dt"]``, ``cell.get("dt")``, a variable named ``dt``, or a variable
    given one of those (``aliases``: ``h = cell["dt"]``)."""
    read: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) and sub.slice.value in names:
            read.add(sub.slice.value)
        elif (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get"
              and sub.args and isinstance(sub.args[0], ast.Constant) and sub.args[0].value in names):
            read.add(sub.args[0].value)
        elif isinstance(sub, ast.Name) and sub.id in names:
            read.add(sub.id)
        elif isinstance(sub, ast.Name) and aliases and sub.id in aliases:
            read.add(aliases[sub.id])
    return read


def _aliases(text: str, names: set[str]) -> dict[str, str]:
    """The variables the code gives a setting's value to (``h = cell["dt"]``), by the setting they hold."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            read = _reads_setting(node.value, names, out)
            if len(read) == 1:
                out[node.targets[0].id] = next(iter(read))
    return out


def _asks_for_a_setting(old: str, new: str, settings: dict[str, list[Any]]) -> list[Any]:
    """The values of a check's settings that a comparison the edit adds tests a setting for (``if cell["dt"] == 0.1``,
    ``if dt == 0.1``): the way to answer a check specially instead of computing it. A comparison of a setting with any
    other value, and one that does not read a setting, is a change like any other."""
    before = {ast.dump(c) for c in _compares(old)}
    aliases = _aliases(new, set(settings))
    found: list[Any] = []
    for compare in _compares(new):
        if ast.dump(compare) in before:
            continue
        for name in _reads_setting(compare, set(settings), aliases):
            for node in ast.walk(compare):
                if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
                    continue
                for value in settings[name]:
                    same = (math.isclose(float(node.value), float(value), rel_tol=1e-12, abs_tol=0.0)
                            if _num(node.value) is not None and _num(value) is not None else node.value == value)
                    if same and value not in found:
                        found.append(value)
    return found


def edited_path(edit: Edit) -> str:
    """The file an edit changes, by its path in ``code/`` (``simulate.py``, ``<package>/model.py``)."""
    name = str(edit.file).replace("\\", "/").strip("/")
    return name[len("code/"):] if name.startswith("code/") else name


def check_edit(files: dict[str, str], edit: Edit, allowed: list[str], *, tried: set[str],
               expected: list[float], measures: set[str] | None = None,
               settings: dict[str, list[Any]] | None = None) -> tuple[str | None, str]:
    """``(the new text of the edited file, "")`` when the edit may be run, else ``(None, why it is refused)``. ``tried``:
    the normalized forms of the versions already run (the first one included). ``expected``: the values the checks
    expect (and the criteria's targets); ``measures``: the names of the numbers the checks read from what the simulation
    returns; ``settings``: the values of each setting the checks run on, by the setting's name."""
    name = edited_path(edit)
    if name not in allowed:
        return None, (f"it changes {edit.file!r}, which is not part of the simulation (only "
                      f"{', '.join(allowed) or 'none'} may be changed)")
    old = files.get(name)
    if old is None:
        return None, f"{name} does not exist"
    if "\r\n" in old and "\r\n" not in edit.find:  # the model saw the file with plain line endings
        edit = Edit(file=edit.file, find=edit.find.replace("\n", "\r\n"),
                    replace=edit.replace.replace("\n", "\r\n"), why=edit.why)
    count = old.count(edit.find)
    if count != 1:
        return None, (f"the text to replace is {'not in' if count == 0 else f'{count} times in'} {name}; it must be "
                      "there exactly once")
    new = old.replace(edit.find, edit.replace, 1)
    if new == old:
        return None, "it changes nothing"
    after = normalized(new)
    if after is None:
        return None, f"{name} would no longer be valid Python"
    if after == normalized(old):
        return None, ("it changes nothing the simulation computes (only comments, layout, docstrings or printed or "
                      "logged text)")
    if _fingerprint({**files, name: new}) in tried:
        return None, "it gives back a version already tried"
    hits = planted(old, new, expected)
    if hits:
        return None, (f"it writes {', '.join(f'{h:g}' for h in hits)} into the code, a value a check expects: a check "
                      "must be passed by computing it, not by writing it in")
    keys = set(measures or ())
    if keys and _measure_exprs(old, keys) != _measure_exprs(new, keys):
        return None, (f"it changes how {', '.join(sorted(keys))} is worked out where the simulation returns it, the "
                      "number a check reads; what a check measures is fixed with the protocol: change what feeds it (the "
                      "body of the function it calls, the step before it) instead")
    asked = _asks_for_a_setting(old, new, dict(settings or {}))
    if asked:
        return None, (f"it adds a comparison with {', '.join(repr(a) for a in asked)}, a setting a check runs on: a "
                      "check must be passed by computing the answer, not by recognising the check")
    return new, ""


def _fingerprint(files: dict[str, str]) -> str:
    parts = [f"{name}\n{normalized(text) or text}" for name, text in sorted(files.items())]
    return hashlib.sha256("\n\0".join(parts).encode("utf-8", "surrogateescape")).hexdigest()


def fingerprint(files: dict[str, str]) -> str:
    """One hash of what a version of the code computes, for "a version already tried"."""
    return _fingerprint(files)


# --- what a round must not change ------------------------------------------------------------------------------------


def _hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def guard_files(quest_root: Path) -> dict[str, Path]:
    """The files FI judges the code by, by the name the log gives them."""
    here = Path(__file__).resolve().parent
    out = {p.as_posix(): Path(quest_root) / p for p in _QUEST_FILES}
    out.update({f"FI's own core/{name}": here / name for name in _ENGINE_FILES})
    return out


#: What git keeps in ``code/.git`` that FI's commits change themselves (never put back or hashed): its objects and refs.
_GIT_OWN = frozenset({"objects", "refs", "logs", "index", "ORIG_HEAD", "COMMIT_EDITMSG"})


def _is_link(path: Path) -> bool:
    """A symbolic link, or a Windows junction (which Python before 3.12 does not call a link)."""
    try:
        if os.path.islink(path):
            return True
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
        return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except OSError:
        return False


def _walk(folder: Path, *, skip: frozenset[str] = frozenset()) -> tuple[list[Path], list[Path]]:
    """``(files, links)`` under ``folder``, never following a link or a junction (a link is listed, not entered), and
    leaving out Python's caches and the top-level names in ``skip``."""
    files: list[Path] = []
    links: list[Path] = []
    if not folder.is_dir() or _is_link(folder):
        return files, links
    for base, dirs, names in os.walk(folder, followlinks=False):
        here = Path(base)
        top = here == folder
        kept = []
        for d in sorted(dirs):
            if d == "__pycache__" or (top and d in skip):
                continue
            if _is_link(here / d):
                links.append(here / d)
            else:
                kept.append(d)
        dirs[:] = kept
        for n in sorted(names):
            if top and n in skip:
                continue
            (links if _is_link(here / n) else files).append(here / n)
    return files, links


def _code_entries(quest_root: Path) -> tuple[list[Path], list[Path]]:
    """Every file and link in ``code/`` (its git history's own objects and refs aside): the code, and git's settings,
    hooks and anything else that would steer FI's next commit."""
    code = Path(quest_root) / "code"
    files, links = _walk(code, skip=frozenset({".git"}))
    git = code / ".git"
    if git.exists() and (_is_link(git) or not git.is_dir()):
        links.append(git)  # history replaced by a file pointing elsewhere, or by a link: a change in itself
    else:
        more_files, more_links = _walk(git, skip=_GIT_OWN)
        files += more_files
        links += more_links
    return files, links


def guard_hashes(quest_root: Path) -> dict[str, str]:
    """The hash of each file FI judges by, of everything in ``code/`` (see :func:`_code_entries`) and of the loop's own
    copies of the kept versions and of the first run's trial record, which are put back later. A link is listed by
    where it points: making one is a change."""
    root = Path(quest_root)
    out = {name: _hash(path) for name, path in guard_files(root).items()}
    code_files, code_links = _code_entries(root)
    own_files, own_links = _walk(root / SNAPSHOTS)
    for p in [*code_files, *own_files]:
        out[p.relative_to(root).as_posix()] = _hash(p)
    for p in [*code_links, *own_links]:
        try:
            target = os.readlink(p)
        except OSError:
            target = "?"
        out[p.relative_to(root).as_posix()] = f"link:{target}"
    return out


def tree_bytes(quest_root: Path) -> dict[str, bytes | None]:
    """``code/`` as it is, byte for byte (see :func:`_code_entries`), to put back after a round whose run changed it. A
    file that cannot be read is kept as ``None``: it is left alone, never deleted."""
    root = Path(quest_root)
    out: dict[str, bytes | None] = {}
    files, _links = _code_entries(root)
    for p in files:
        try:
            out[p.relative_to(root).as_posix()] = p.read_bytes()
        except OSError:
            out[p.relative_to(root).as_posix()] = None
    return out


def _remove_link(path: Path) -> None:
    try:
        if path.is_dir() and not os.path.islink(path):
            os.rmdir(path)  # a junction: removes the junction itself, never what it points at
        else:
            os.unlink(path)
    except OSError:
        pass


def put_back_tree(quest_root: Path, saved: dict[str, bytes | None]) -> bool:
    """``code/`` back to ``saved``: a link the run made is removed (the link itself, never what it points at), a file
    it added is removed, one it changed or removed is written back. ``False`` when ``code/.git`` is no longer a folder
    (the history was replaced): FI then makes no commit into it."""
    root = Path(quest_root)
    code = root / "code"
    if _is_link(code):
        return False  # code/ itself made a link: nothing is written through it
    files, links = _code_entries(root)
    git = code / ".git"
    had_history = any(rel.startswith("code/.git/") for rel in saved)
    history_ok = (git.is_dir() and not _is_link(git)) if had_history else not git.exists()
    for link in links:
        _remove_link(link)  # code/.git made a link or a file included: removed, never written through
    for p in files:
        if p.relative_to(root).as_posix() not in saved:
            try:
                p.unlink()
            except OSError:
                pass
    for rel, data in saved.items():
        if data is None or (not history_ok and rel.startswith("code/.git/")):
            continue  # a history the run replaced or deleted is not rebuilt from its settings alone
        path = root / rel
        try:
            if _is_link(path):
                _remove_link(path)
            if not path.is_file() or path.read_bytes() != data:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        except OSError:
            pass
    return history_ok


def guard_bytes(quest_root: Path) -> dict[str, bytes | None]:
    """The quest's own records as they are, to put back after a round that changed them."""
    out: dict[str, bytes | None] = {}
    for rel in _QUEST_FILES:
        try:
            out[rel.as_posix()] = (Path(quest_root) / rel).read_bytes()
        except OSError:
            out[rel.as_posix()] = None
    return out


def put_back(quest_root: Path, saved: dict[str, bytes | None]) -> None:
    for rel, data in saved.items():
        path = Path(quest_root) / rel
        try:
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        except OSError:
            pass


def changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


# --- the ratchet -----------------------------------------------------------------------------------------------------


def _badness(row: dict[str, Any], value: float) -> float:
    """Lower is better, whatever the criterion's direction."""
    direction = row.get("direction")
    if direction == "higher":
        return -value
    if direction == "target":
        target = _num(row.get("target"))
        return abs(value - target) if target is not None else value
    return value


def compare(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each criterion FI counts, compared on its own between two versions: ``change`` is ``better`` or ``worse`` (by more
    than the criterion's own tolerance), ``same``, ``gained`` (measured now, not before), ``lost`` (measured before, not
    now) or ``unmeasured``."""
    b = {r["name"]: r for r in before if isinstance(r, dict) and r.get("counts")}
    a = {r["name"]: r for r in after if isinstance(r, dict)}
    out: list[dict[str, Any]] = []
    for name in [*b, *[n for n, r in a.items() if r.get("counts") and n not in b]]:
        rb, ra = b.get(name) or {}, a.get(name) or {}
        row = rb or ra
        vb = _num(rb.get("value")) if rb else None
        va = _num(ra.get("value")) if ra.get("counts") else None
        tol = abs(_num(row.get("tolerance")) or 0.0)
        if vb is None and va is None:
            change = "unmeasured"
        elif vb is None:
            # Measured now and not before: better only when it meets its bar; otherwise nothing to compare it with.
            change = "gained" if ra.get("met") is True else "same"
        elif va is None:
            change = "lost"
        else:
            db, da = _badness(row, vb), _badness(row, va)
            change = "better" if da < db - tol else "worse" if da > db + tol else "same"
        out.append({"name": name, "before": vb, "after": va, "tolerance": tol, "change": change,
                    "met": ra.get("met") if ra else None})
    return out


def verdict(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    """``better`` and ``worse`` (the criteria's names), ``broken`` (the change stopped every criterion measured before from
    being measured: the simulation no longer runs its checks) and ``best`` (kept: at least one better, none worse)."""
    better = [c["name"] for c in comparison if c["change"] in ("better", "gained")]
    worse = [c["name"] for c in comparison if c["change"] in ("worse", "lost")]
    measured_before = [c for c in comparison if c["before"] is not None]
    broken = bool(measured_before) and all(c["change"] == "lost" for c in measured_before)
    return {"better": better, "worse": [] if broken else worse, "broken": broken,
            "best": bool(better) and not worse and not broken}


def counted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if isinstance(r, dict) and r.get("counts")]


def all_met(rows: list[dict[str, Any]]) -> bool:
    """Every criterion FI counts is met (one with no bar set is never met)."""
    mine = counted(rows)
    return bool(mine) and all(r.get("met") is True for r in mine)


def values_line(rows: list[dict[str, Any]], comparison: list[dict[str, Any]] | None = None) -> str:
    """``name: value (met / not met, better / worse / same)`` for each criterion FI counts."""
    by = {c["name"]: c for c in comparison or []}
    parts = []
    for r in counted(rows) or [r for r in rows if isinstance(r, dict)]:
        state = {True: "met", False: "not met", None: "no bar"}[r.get("met")]
        change = by.get(r["name"], {}).get("change")
        parts.append(f"{r['name']}: {_fmt(r.get('value'))} ({state}{', ' + change if change else ''})")
    return "; ".join(parts) or "no criterion measured"


# --- the trials a round runs -----------------------------------------------------------------------------------------


def series_of(run: Any, names: set[str] | list[str]) -> dict[str, dict[str, list[float]]]:
    """Each trial number's values per setting, in trial order, from a :class:`core.trial_runner.TrialRun`."""
    out: dict[str, dict[str, list[float]]] = {}
    for cell in getattr(run, "cells", None) or []:
        rows = sorted((r for r in cell.rows if r.get("status") == "ok"), key=lambda r: r.get("trial") or 0)
        for name in names:
            values = [v for v in ((r.get("values") or {}).get(name) for r in rows) if _num(v) is not None]
            if values:
                out.setdefault(name, {})[cell.key] = [float(v) for v in values]
    return out


# --- the prompt ------------------------------------------------------------------------------------------------------


def criteria_block(items: list[dict[str, Any]], protocol: dict[str, Any] | None) -> str:
    """Each criterion FI counts in a sentence, with the known-answer check it rests on (its case, what it expects and
    where that comes from). The protocol's metrics, contrasts and precision target are never shown."""
    from . import criteria as _criteria
    from . import oracle_check as _oracle

    oracles = {str(o["name"]).strip(): o for o in _oracle.declared(protocol)}
    lines = []
    for c in items:
        lines.append(f"- {_criteria.describe(c, protocol)}")
        oracle = oracles.get(str(c.get("oracle") or ""))
        if oracle is not None:
            case = _oracle.case_of(oracle)
            detail = [f"the check: {str(oracle.get('check') or '').strip() or '(not described)'}"]
            if case is not None:
                detail.append(f"FI runs the simulation on the setting {json.dumps(case[0], default=str)} and reads "
                              f"`{case[1]}` from what it returns")
            detail.append(f"expected {oracle.get('expected')!r} within {oracle.get('tolerance')!r}")
            if oracle.get("reference"):
                detail.append(f"from: {' '.join(str(oracle['reference']).split())[:400]}")
            lines.append("  (" + "; ".join(detail) + ")")
    return "\n".join(lines) or "(none)"


def history_block(rounds: list[dict[str, Any]]) -> str:
    if not rounds:
        return "(none yet: this is the first change)"
    out = []
    for r in rounds:
        head = f"- round {r['round']}: {r.get('why') or ''} -> {r.get('outcome')}"
        if r.get("reason"):
            head += f" ({r['reason']})"
        out.append(head)
        if r.get("values"):
            out.append(f"  values: {r['values']}")
        if r.get("diff"):
            out.append("  change:\n" + "\n".join("    " + line for line in r["diff"].splitlines()[:40]))
    return "\n".join(out)


def code_block(files: dict[str, str], allowed: list[str]) -> str:
    out = []
    for name in allowed:
        text = files.get(name, "").encode("utf-8", "surrogateescape").decode("utf-8", "replace").replace("\r\n", "\n")
        if len(text) > _CODE_CAP:
            text = text[:_CODE_CAP] + f"\n# ... ({len(text) - _CODE_CAP} more characters not shown)"
        out.append(f"### code/{name}\n\n```python\n{text}\n```")
    return "\n\n".join(out)


def diff_text(edit: Edit) -> str:
    def cut(text: str) -> str:
        return text if len(text) <= 1200 else text[:1200] + "..."

    return f"in {edit.file}, replaced:\n{cut(edit.find)}\nwith:\n{cut(edit.replace)}"


# --- the record ------------------------------------------------------------------------------------------------------


def load(quest_root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(quest_root: Path, record: dict[str, Any]) -> bool:
    """Write the loop's record; ``False`` when it could not be written (the caller says so in the log)."""
    path = Path(quest_root) / RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, default=str, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        return True
    except (OSError, ValueError):
        return False  # a record that cannot be written never stops a quest


def headline_digest(result_json: Any) -> dict[str, str]:
    """One hash per top-level number or block of the study's results: enough to say a result changed, never its value."""
    if not isinstance(result_json, dict):
        return {}
    return {str(k): hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            for k, v in result_json.items() if not str(k).startswith("_")}


def summary(record: dict[str, Any]) -> str:
    """One plain sentence of what the loop did: rounds, why it stopped, which version was kept."""
    rounds = [r for r in record.get("rounds") or [] if isinstance(r, dict)]
    if not record.get("stopped"):
        return ""
    ran = sum(1 for r in rounds if r.get("ran"))
    kept = int(record.get("best_round") or 0)
    text = (f"FI tried {len(rounds)} change(s) to the simulation, one at a time ({ran} run and measured), and stopped "
            f"because {record['stopped']}. ")
    if kept:
        text += (f"The version from round {kept} was kept: the last change that made a check of correctness better "
                 "without making another worse beyond its tolerance.")
    elif record.get("stopped") == STOP_CUT_SHORT:
        text += "Nothing it changed was kept."
    elif not record.get("fell_back"):
        text += "The first version was kept: no change made a check of correctness better without making another worse."
    worse = [r for r in rounds if r.get("worse")]
    if worse:
        text += " " + "; ".join(f"round {r['round']} made {', '.join(r['worse'])} worse and was not kept" for r in worse) + "."
    if record.get("fell_back"):
        source = record["fell_back"] if isinstance(record["fell_back"], str) else "a later round"
        which = f"The version from {source}" if source.startswith(("round", "a later")) else f"The version {source}"
        text += (f" {which} was kept at first, but its full run produced no result, so the first version was put back "
                 "and is the one reported.")
    earlier = [e for e in record.get("earlier") or [] if isinstance(e, dict)]
    if earlier:
        text += f" (Before this, {len(earlier)} earlier pass(es) of the loop in this quest: " + " ".join(
            summary({k: v for k, v in e.items() if k != "earlier"}) or "cut short." for e in earlier) + ")"
    return text


def write_note(quest_root: Path) -> str:
    """What the writer is told about the loop (appended to the evidence note), or an empty string when it did not run."""
    record = load(quest_root)
    # Only a loop whose results are the ones being written up: one over code that was replaced since (a rerun from an
    # earlier step, a redesign) is not what produced them.
    from . import criteria as _criteria

    runs = [r for r in _criteria.history(quest_root) if not r.get("improve")]
    if record.get("result_n") is None or not runs or runs[-1].get("n") != record.get("result_n"):
        return ""
    text = summary(record)
    if not text:
        return ""
    return ("The simulation was improved against its checks of correctness before these results were produced. State "
            "this in the methods in one or two sentences, in these terms: " + text + " The version was chosen by the "
            "checks of correctness only, never by the study's own results; the results you are given come from one "
            "full run of the version kept.")


_LINE = re.compile(r"\s+")


def one_line(text: str, limit: int = 200) -> str:
    text = _LINE.sub(" ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."
