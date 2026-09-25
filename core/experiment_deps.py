"""What an experiment's quest environment is given before it runs: its packages, and the skills it was offered.

The code-writing step returns the script and a list of packages to install. Three things used to go wrong between
that list and a run, and all three looked like a broken environment:

- One name pip cannot find (a skill's name, the script's own helper file, a package that does not exist) failed the
  whole ``pip install`` line, so numpy and everything else on it was not installed either, and the run then died on
  ``import numpy``. A failed line is now retried one package at a time, so one bad name costs only itself; the
  script's own files are left out, and a skill's name that pip cannot install is explained as the skill.
- A skill's own packages (``pip_requires``, recorded when it was imported) went only into FI's interpreter, which a
  quest's own environment (the research profile's clean venv) cannot see. They are now installed into the quest's.
- A library skill's folder was never on the experiment's ``PYTHONPATH``; its self-test (run on FI's interpreter with
  the folder on the path) passed while the experiment could not import it. It is now put on the path.

What still cannot be installed is written as a note the repair steps read before the run's own error.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

# The distribution name at the start of a requirement ("numpy>=1.26", "scikit-learn[all]", "pkg ; python_version>..").
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize(name: str) -> str:
    """PEP 503 form: lower case, runs of ``-``, ``_`` and ``.`` as one ``-`` (so ``lieflat_charts`` is ``lieflat-charts``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(dep: str) -> str:
    """The distribution name of a requirement string, or ``""`` when it has none (a URL, a path)."""
    m = _NAME_RE.match(str(dep))
    return m.group(1) if m and "/" not in str(dep) and "\\" not in str(dep) else ""


def split_deps(
    deps: Iterable[str], *, skills: Iterable[Any] = (), local_modules: Iterable[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split the requested packages into those to install and those left out, each with the reason.

    Left out before pip is asked: only a module the quest's own ``code/`` folder holds (it shadows any package of that
    name anyway). A selected skill's name is still asked of pip: many skills are named after the package they teach
    (matplotlib, seaborn, scipy), whatever their kind, and only pip knows whether a name is a package. When pip cannot
    install one, ``explain_failures`` says how the skill is used instead. Repeats are dropped; order is kept."""
    local = {normalize(m) for m in local_modules}
    install: list[str] = []
    dropped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for dep in deps:
        dep = str(dep).strip()
        name = normalize(requirement_name(dep) or dep)
        if not dep or name in seen:
            continue
        seen.add(name)
        if name in local:
            dropped.append((dep, f"is the quest's own file code/{name.replace('-', '_')}.py, not a package"))
        else:
            install.append(dep)
    return install, dropped


def explain_failures(
    failed: list[tuple[str, str]], skills: Iterable[Any],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split pip's failures into ``(skills, others)``: a name pip could not install that is a selected skill's gets
    how that skill is used (a tool is run, a library is imported from the path), not "remove the import"."""
    by_skill = {normalize(s.name): s for s in skills}
    as_skill: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []
    for dep, why in failed:
        skill = by_skill.get(normalize(requirement_name(dep) or dep))
        if skill is not None:
            as_skill.append((dep, skill_hint(skill)))
        else:
            others.append((dep, why))
    return as_skill, others


def _kind(skill: Any) -> str:
    return str(getattr(getattr(skill, "kind", None), "value", getattr(skill, "kind", ""))).lower()


def _is_library(skill: Any) -> bool:
    return _kind(skill) == "library"


def skill_hint(skill: Any) -> str:
    """How a skill is used, for a name that was asked of pip."""
    if _is_library(skill):
        return (f"is the skill {skill.name}, whose folder FI puts on the experiment's path: import it from there as "
                f"its API surface says; it is not a PyPI package")
    return (f"is the skill {skill.name}, a tool, not a Python package: it cannot be imported or pip-installed; run its "
            f"scripts as its instructions say")


def skill_requirements(skills: Iterable[Any]) -> list[str]:
    """The packages the selected skills recorded as needing (``pip_requires`` in their provenance)."""
    out: list[str] = []
    for s in skills:
        try:
            declared = s.provenance().get("pip_requires")
        except Exception:  # noqa: BLE001 -- unreadable provenance is "none declared"
            declared = None
        if isinstance(declared, list):
            out.extend(str(x).strip() for x in declared if str(x).strip())
    return list(dict.fromkeys(out))


def library_paths(skills: Iterable[Any]) -> list[Path]:
    """The folders a library skill's code is imported from: the skill's folder, and its ``scripts/`` when it has one."""
    out: list[Path] = []
    for s in skills:
        if not _is_library(s):
            continue
        out.append(Path(s.path))
        if (Path(s.path) / "scripts").is_dir():
            out.append(Path(s.path) / "scripts")
    return list(dict.fromkeys(out))


def failure_reason(stderr: str) -> str:
    """One plain line for why pip could not install a single package."""
    text = stderr or ""
    if "No matching distribution" in text or "Could not find a version that satisfies" in text:
        return "no such package on PyPI (or none for this Python version)"
    if "ResolutionImpossible" in text or "conflict" in text.lower():
        return "its version requirements conflict with the other packages"
    if "Network is unreachable" in text or "NewConnectionError" in text or "Temporary failure in name resolution" in text:
        return "PyPI could not be reached"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else "pip failed")[:200]


def repair_note(
    dropped: list[tuple[str, str]], failed: list[tuple[str, str]], unusable_skills: Iterable[str] = (),
) -> str:
    """The note the repair steps read first: what was not installed and why, and selected skills that cannot be used."""
    unusable = list(unusable_skills)
    if not dropped and not failed and not unusable:
        return ""
    lines = ["FI NOTE (packages): these requested packages were not installed:"] if dropped or failed else []
    lines += [f"- {dep}: {why}" for dep, why in dropped]
    lines += [f"- {dep}: {why}; remove the import, or use a package that exists" for dep, why in failed]
    if unusable:
        lines.append("FI NOTE (skills): selected for this quest but not usable now (see run.log), so do not import "
                     "or run them: " + ", ".join(unusable))
    return "\n".join(lines) + "\n"
