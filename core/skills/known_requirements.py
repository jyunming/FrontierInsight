"""The packages each preset skill needs, and the versions a skill must not get.

A skill records what it needs in ``pip_requires`` (provenance.json) when it is imported. Skills imported before that
record existed carry none, and a quest then installed nothing for them: fine while every experiment ran on FI's own
interpreter (where the packages were), a failure in a quest's own clean environment. This table is the fallback for
the preset skills ``scripts/import_scientist_skills.py`` imports, and that script reads its package lists from here, so
the two cannot drift. ``PINS`` holds versions known to break a skill: landlab 2.11.0 declares Python >= 3.11 but uses
3.12-only syntax, so ``import landlab`` fails under 3.11 with a SyntaxError; 2.10.1 imports cleanly.
"""

from __future__ import annotations

import re
import sys
from typing import Any

PRESET_PIP_REQUIRES: dict[str, list[str]] = {
    "pymc": ["pymc", "arviz"],
    "astropy": ["astropy"],
    "rdkit": ["rdkit"],
    "biopython": ["biopython"],
    "pymatgen": ["pymatgen"],
    "scikit-bio": ["scikit-bio"],
    "geopandas": ["geopandas"],
    "cobrapy": ["cobra"],
    "qutip": ["qutip"],
    "scientific-critical-thinking": [],
    "ontology-term-resolution": [],
    "molecular-dynamics": [],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "scientific-visualization": ["matplotlib", "seaborn"],
    "analyze-fasta": [],
    "genome-compare": [],
    "variant-annotation": [],
    "vcf-annotator": [],
    "phylogenetics-builder": [],
    "methylation-clock": [],
    "obspy": ["obspy"],
    "lasio": ["lasio"],
    "welly": ["welly"],
    "gempy": ["gempy"],
    "simpeg": ["simpeg", "discretize"],
    "harmonica": ["harmonica"],
    "landlab": ["landlab==2.10.1"],
    "pastas": ["pastas"],
    "segyio": ["segyio"],
    "xarray": ["xarray"],
    "convergence-study": [],
    "differentiation-schemes": [],
    "linear-solvers": [],
    "mesh-generation": [],
    "nonlinear-solvers": [],
    "numerical-integration": [],
    "numerical-stability": [],
    "time-stepping": [],
    "benchmark-and-mms-planner": [],
    "scikit-image": ["scikit-image"],
    "scipy": ["scipy"],
    "dowhy": ["dowhy"],
    "xgboost-lightgbm": ["xgboost", "lightgbm"],
    "lifelines": ["lifelines"],
    "control-systems": [],
    "state-space-control": [],
    "pid-controller": [],
    "lqr-control": [],
    "kalman-filter": [],
    "weibull-analysis": [],
    "reliability-engineering": [],
    "accelerated-life-testing": [],
    "system-reliability": [],
    "structural-analysis": [],
    "truss-analysis": [],
    "beam-solver": [],
    "fea-fundamentals": [],
    "column-buckling": [],
    "topology-optimization": [],
    "metpy": ["metpy"],
    "map": ["cartopy"],
    "regrid": [],
    "animate": [],
    "neurokit2": ["neurokit2"],
    "causal-inference": [],
    "amortized-workflow": [],
    "mne-python": ["mne"],
    "brian2": ["brian2"],
    "nilearn": ["nilearn"],
    "spikeinterface": ["spikeinterface"],
}

#: A requirement with no version that is known to install a broken one: the pin to use, and the Python versions it is
#: needed on (below the given version). landlab 2.11.0 uses 3.12-only syntax, so only Python < 3.12 needs 2.10.1.
PINS: dict[str, tuple[str, tuple[int, int]]] = {
    "landlab": ("==2.10.1", (3, 12)),
}


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse(requirement: str) -> tuple[str, str, bool]:
    """``(name, extras + specifier + marker text, has_version)`` of a requirement string."""
    try:
        from packaging.requirements import Requirement

        req = Requirement(requirement)
        extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
        marker = f"; {req.marker}" if req.marker else ""
        return req.name, extras + marker, bool(str(req.specifier)) or bool(req.url)
    except Exception:  # noqa: BLE001 -- packaging missing, or an odd string: a plain parse
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)", requirement)
        if not m:
            return requirement.strip(), "", True
        rest = m.group(3).strip()
        return m.group(1), m.group(2) or "", bool(rest) and not rest.startswith(";")


def pinned(requirement: str, *, python: tuple[int, int] | None = None) -> str:
    """``requirement`` with a known-bad unversioned package pinned (extras and markers kept). A requirement that names
    a version is the person's choice and is kept; so is any requirement on a Python the pin is not needed on."""
    requirement = requirement.strip()
    name, extras, has_version = _parse(requirement)
    pin = PINS.get(_normal(name))
    if pin is None or has_version:
        return requirement
    version, below = pin
    if (python or sys.version_info[:2]) >= below:
        return requirement
    extras_only, _, marker = extras.partition(";")
    return f"{name}{extras_only.strip()}{version}" + (f"; {marker.strip()}" if marker.strip() else "")


def requirement_key(requirement: str) -> str:
    """The package a requirement names, normalized (for telling two requirements of one package apart)."""
    return _normal(_parse(requirement)[0])


def pip_requires(skill: Any) -> list[str]:
    """What a skill needs installed: its recorded ``pip_requires``; for a preset skill imported before provenance
    recorded one (its source is the import script's ``skill-sources`` cache), the preset table's entry. Each pinned.
    A skill of one's own that happens to share a preset's name gets nothing it did not declare."""
    try:
        provenance = skill.provenance()
    except Exception:  # noqa: BLE001 -- unreadable provenance is "none recorded"
        provenance = {}
    declared = provenance.get("pip_requires") if isinstance(provenance, dict) else None
    if not isinstance(declared, list):
        source = str((provenance or {}).get("imported_from") or "") if isinstance(provenance, dict) else ""
        preset = "skill-sources" in source.replace("\\", "/")
        declared = PRESET_PIP_REQUIRES.get(getattr(skill, "name", ""), []) if preset else []
    return list(dict.fromkeys(pinned(str(x)) for x in declared if str(x).strip()))
