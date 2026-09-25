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

#: A bare requirement that is known to install a broken version, and what to install instead.
PINS: dict[str, str] = {
    "landlab": "landlab==2.10.1",
}


def pinned(requirement: str) -> str:
    """``requirement`` with a known-bad bare name replaced by its pin (a requirement that names a version is kept)."""
    bare = requirement.strip()
    key = re.sub(r"[-_.]+", "-", bare).lower()
    return PINS.get(key, bare) if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", bare) else bare


def pip_requires(skill: Any) -> list[str]:
    """What a skill needs installed: its recorded ``pip_requires``, else the preset table's entry, each pinned."""
    try:
        declared = skill.provenance().get("pip_requires")
    except Exception:  # noqa: BLE001 -- unreadable provenance is "none recorded"
        declared = None
    if not isinstance(declared, list):
        declared = PRESET_PIP_REQUIRES.get(getattr(skill, "name", ""), [])
    return list(dict.fromkeys(pinned(str(x)) for x in declared if str(x).strip()))
