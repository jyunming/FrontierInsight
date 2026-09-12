"""requirements.txt and pyproject.toml must declare the same packages.

FI carries two dependency manifests. `requirements.txt` is what a human
installs; `pyproject.toml` is what `pip install -e .[dev,docker]` installs,
which is what CI runs. They drifted once: the PPTX renderer's `python-pptx`
was added to requirements.txt only, so every developer with a globally
installed python-pptx saw a green local suite while CI failed on both
platforms with `No module named 'pptx'`.

A missing declaration is invisible in exactly the environment that would
catch it, because the package is usually already present for some other
reason. This test makes the drift itself the failure.

The hard direction is requirements -> pyproject: a package a user is told to
install but that the packaged distribution never declares is a broken
`pip install frontier-insight`. The reverse direction is checked too, since a
dependency the packaged install pulls but requirements.txt omits misleads
anyone setting up from that file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — requires-python is >=3.11
    tomllib = None

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"
REQUIREMENTS = REPO / "requirements.txt"


def _canon(name: str) -> str:
    """PEP 503 normalisation: pypandoc_binary and pypandoc-binary are one
    package, and `python-pptx` must not be confused with its `pptx` module."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _split(spec: str) -> tuple[str, str]:
    """`axon-rag>=0.4.2` -> (`axon-rag`, `>=0.4.2`). Drops extras markers,
    environment markers and trailing comments."""
    spec = spec.split(";", 1)[0]           # environment marker
    spec = spec.split("#", 1)[0]           # trailing comment
    spec = spec.strip()
    m = re.match(r"^([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(.*)$", spec)
    if not m:
        return _canon(spec), ""
    return _canon(m.group(1)), m.group(2).replace(" ", "")


def _name_of(spec: str) -> str:
    return _split(spec)[0]


def _requirements_names() -> set[str]:
    names: set[str] = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        # Commented-out optional deps (e.g. `# scholarly>=1.7`) are
        # deliberately not installed, so they are not declarations.
        if not line or line.startswith(("#", "-")):
            continue
        name = _name_of(line)
        if name:
            names.add(name)
    return names


def _pyproject_names() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    return {_name_of(s) for s in specs}


@pytest.mark.skipif(tomllib is None, reason="needs tomllib (3.11+)")
def test_every_requirement_is_declared_in_pyproject() -> None:
    missing = _requirements_names() - _pyproject_names()
    assert not missing, (
        f"In requirements.txt but NOT declared in pyproject.toml: "
        f"{sorted(missing)}. CI installs `pip install -e .[dev,docker]`, so "
        f"these are absent there and from `pip install frontier-insight` — "
        f"they will only appear to work on a machine that happens to have "
        f"them already. Add them to [project] dependencies or an extra."
    )


@pytest.mark.skipif(tomllib is None, reason="needs tomllib (3.11+)")
def test_every_pyproject_dependency_is_listed_in_requirements() -> None:
    missing = _pyproject_names() - _requirements_names()
    assert not missing, (
        f"Declared in pyproject.toml but absent from requirements.txt: "
        f"{sorted(missing)}. Anyone setting up with "
        f"`pip install -r requirements.txt` gets an incomplete environment."
    )


def _requirements_specs() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        name, ver = _split(line)
        if name:
            out.setdefault(name, set()).add(ver)
    return out


def _pyproject_specs() -> dict[str, set[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    out: dict[str, set[str]] = {}
    groups = [project.get("dependencies", [])]
    groups += list(project.get("optional-dependencies", {}).values())
    for group in groups:
        for spec in group:
            name, ver = _split(spec)
            out.setdefault(name, set()).add(ver)
    return out


@pytest.mark.skipif(tomllib is None, reason="needs tomllib (3.11+)")
def test_pyproject_does_not_contradict_itself() -> None:
    """The `all` extra repeats what the individual extras declare (a
    deliberate choice — see its comment in pyproject.toml), which is exactly
    where a floor gets bumped in one place and not the other."""
    conflicting = {n: sorted(v) for n, v in _pyproject_specs().items()
                   if len(v) > 1}
    assert not conflicting, (
        f"pyproject.toml declares different version specifiers for the same "
        f"package: {conflicting}. Which one applies depends on which extras "
        f"the user selects."
    )


@pytest.mark.skipif(tomllib is None, reason="needs tomllib (3.11+)")
def test_version_floors_agree_across_manifests() -> None:
    """A name declared in both files but at different floors gives two users
    following two documented install paths two different environments —
    the drift that let `axon-rag>=0.4.2` sit beside `axon-rag>=0.4`."""
    req, proj = _requirements_specs(), _pyproject_specs()
    mismatched = {
        n: {"requirements.txt": sorted(req[n]), "pyproject.toml": sorted(proj[n])}
        for n in req.keys() & proj.keys()
        if req[n] != proj[n]
    }
    assert not mismatched, (
        f"Version specifier drift between the two manifests: {mismatched}. "
        f"Bump both together."
    )
