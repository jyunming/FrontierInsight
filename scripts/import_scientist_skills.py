"""Reproduce this project's curated scientist-workflow skill set from scratch.

FI ships no skills — the discovery root is user state
(``~/.frontier-insight/skills``, or ``FI_SKILLS_DIR``), not repository
content, so a fresh clone of this repo starts with an empty skill library
(see ``core/skills/registry.py``). The skills below were sourced and
reviewed across two working sessions; this script reproduces the
*sourcing* step so nobody has to re-derive eleven source repos, seventy-odd
folder names, and the command sequence by hand. It does NOT approve
anything for you — approval binds to a person's judgement
(``--approve-skill --approve-as WHO``) and is deliberately not a thing a
script gets to do.

Sources (repo -> what it contributes):
  * K-Dense-AI/scientific-agent-skills — Bayesian modeling, astronomy,
    cheminformatics, bioinformatics, materials science, geospatial,
    metabolic modeling, quantum simulation, ontology-term resolution,
    molecular dynamics. Already behind four skills trusted in this
    library (scikit-learn, get-available-resources,
    iso-standards-readiness, what-if-oracle), so format/quality are a
    known quantity.
  * larashero3-dotcom/lieflat-charts — template-driven charts/reports,
    with a small FI-authored addition bundled alongside this script
    (``scripts/skill-assets/lieflat-charts/``): a static-render step,
    since FI's paper_pdf/poster/slides outputs can't embed a live JS
    chart.
  * ClawBio/ClawBio — genomics/bioinformatics capability skills. 96
    skills total upstream; only a small, policy-clean, non-literature
    subset is imported here (see SKILLS below) — most of the rest are
    literature-retrieval, MCP-service wrappers, or narrow pipeline
    integrations, deliberately left out.
  * SteadfastAsArt/geoscience-skills — seismology, well logs, 3D geo
    modeling, geophysical inversion, hydrology.
  * HeshamFS/materials-simulation-skills — numerical-methods
    methodology (convergence, meshing, solvers, stability,
    verification & validation) — general enough to matter beyond
    materials specifically.
  * tondevrel/scientific-agent-skills — a handful confirmed NOT to
    duplicate K-Dense's catalogue: image analysis, general SciPy,
    causal inference, gradient boosting, survival analysis.
  * MP-AI-20/mechanical-engineering-skills — control systems,
    reliability engineering, structural/FEA — three domains FI had
    zero coverage of.
  * Z-Richard/Atmos-sci-skills — atmospheric/climate science.
  * jaechang-hits/SciAgent-Skills — physiological signal processing
    (neurokit2) specifically; the rest of this repo's broad
    life-science scope overlaps K-Dense's.
  * Learning-Bayesian-Statistics/baygent-skills — causal inference and
    simulation-based inference; its base Bayesian-workflow skill is
    skipped as redundant with the already-imported pymc skill.
  * HughYau/neuroforge-skills — neuroscience: MEG/EEG, spiking network
    simulation, neuroimaging ML, spike-sorting.

What this does, per skill:
  1. Clone (or refresh an already-cloned) source repo into ``--cache-dir``.
  2. Import via ``core.skills.importer`` — lands UNTESTED; a self-test is
     auto-generated if the source didn't ship one, same as any import.
  3. Run the static scan (``core.skills.scan``) and report the summary.
  4. Print the exact approve command (bulk and per-skill). Never runs it.

Usage:
    python scripts/import_scientist_skills.py
    python scripts/import_scientist_skills.py --cache-dir /path/to/cache
    python scripts/import_scientist_skills.py --skip pymc,rdkit
    python scripts/import_scientist_skills.py --pip-install

``--pip-install`` additionally installs the underlying Python packages
(pymc, astropy, rdkit, ...) into *this* interpreter's environment — a real,
visible side effect on what may be a shared install. Off by default; either
way, the exact pip line is printed so you can run it yourself. Not every
skill below maps to one clean pip package (several are methodology/
knowledge skills rather than a library wrapper) — those are simply omitted
from the pip line; check the skill's own SKILL.md compatibility section.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPOS: dict[str, str] = {
    "kdense": "https://github.com/K-Dense-AI/scientific-agent-skills.git",
    "lieflat": "https://github.com/larashero3-dotcom/lieflat-charts.git",
    "clawbio": "https://github.com/ClawBio/ClawBio.git",
    "geoscience": "https://github.com/SteadfastAsArt/geoscience-skills.git",
    "matsim": "https://github.com/HeshamFS/materials-simulation-skills.git",
    "tondevrel": "https://github.com/tondevrel/scientific-agent-skills.git",
    "mecheng": "https://github.com/MP-AI-20/mechanical-engineering-skills.git",
    "atmos": "https://github.com/Z-Richard/Atmos-sci-skills.git",
    "sciagent": "https://github.com/jaechang-hits/SciAgent-Skills.git",
    "baygent": "https://github.com/Learning-Bayesian-Statistics/baygent-skills.git",
    "neuroforge": "https://github.com/HughYau/neuroforge-skills.git",
}

# name -> (repo key, relative path inside that clone, pip packages, needs --despite-findings)
SKILLS: dict[str, tuple[str, str, list[str], bool]] = {
    # -- K-Dense-AI/scientific-agent-skills --
    "pymc": ("kdense", "skills/pymc", ["pymc", "arviz"], False),
    "astropy": ("kdense", "skills/astropy", ["astropy"], False),
    "rdkit": ("kdense", "skills/rdkit", ["rdkit"], False),
    "biopython": ("kdense", "skills/biopython", ["biopython"], False),
    "pymatgen": ("kdense", "skills/pymatgen", ["pymatgen"], False),
    "scikit-bio": ("kdense", "skills/scikit-bio", ["scikit-bio"], False),
    "geopandas": ("kdense", "skills/geopandas", ["geopandas"], False),
    "cobrapy": ("kdense", "skills/cobrapy", ["cobra"], False),
    "qutip": ("kdense", "skills/qutip", ["qutip"], False),
    "scientific-critical-thinking": ("kdense", "skills/scientific-critical-thinking", [], False),
    "ontology-term-resolution": ("kdense", "skills/ontology-term-resolution", [], True),
    "molecular-dynamics": ("kdense", "skills/molecular-dynamics", [], False),
    # -- larashero3-dotcom/lieflat-charts (path "" -> repo root, see _prepare_lieflat) --
    "lieflat-charts": ("lieflat", "", [], False),
    # -- ClawBio/ClawBio (curated subset -- see module docstring) --
    "analyze-fasta": ("clawbio", "skills/analyze-fasta", [], False),
    "genome-compare": ("clawbio", "skills/genome-compare", [], False),
    "variant-annotation": ("clawbio", "skills/variant-annotation", [], False),
    "vcf-annotator": ("clawbio", "skills/vcf-annotator", [], False),
    "phylogenetics-builder": ("clawbio", "skills/phylogenetics-builder", [], False),
    "methylation-clock": ("clawbio", "skills/methylation-clock", [], False),
    # -- SteadfastAsArt/geoscience-skills (flat at repo root, not under skills/) --
    "obspy": ("geoscience", "obspy", ["obspy"], False),
    "lasio": ("geoscience", "lasio", ["lasio"], False),
    "welly": ("geoscience", "welly", ["welly"], False),
    "gempy": ("geoscience", "gempy", ["gempy"], False),
    # simpeg's bundled scripts import discretize directly (mesh generation),
    # not just simpeg's own re-exports -- pip-installing simpeg alone leaves
    # `from discretize import TensorMesh` unresolved.
    "simpeg": ("geoscience", "simpeg", ["simpeg", "discretize"], False),
    "harmonica": ("geoscience", "harmonica", ["harmonica"], False),
    # Pinned: landlab 2.11.0's own code uses Python 3.12+ generic-function
    # syntax (`def f[T](...)`) despite the package claiming
    # `requires_python: >=3.11` on PyPI -- a real upstream metadata bug,
    # verified live (2.11.0 fails a bare `import landlab` with SyntaxError
    # under 3.11; 2.10.1 imports cleanly). Re-check this pin once landlab
    # either fixes the classifier or 3.11 support is genuinely dropped.
    "landlab": ("geoscience", "landlab", ["landlab==2.10.1"], False),
    "pastas": ("geoscience", "pastas", ["pastas"], False),
    "segyio": ("geoscience", "segyio", ["segyio"], False),
    "xarray": ("geoscience", "xarray", ["xarray"], False),
    # -- HeshamFS/materials-simulation-skills --
    "convergence-study": ("matsim", "skills/core-numerical/convergence-study", [], False),
    "differentiation-schemes": ("matsim", "skills/core-numerical/differentiation-schemes", [], False),
    "linear-solvers": ("matsim", "skills/core-numerical/linear-solvers", [], False),
    "mesh-generation": ("matsim", "skills/core-numerical/mesh-generation", [], False),
    "nonlinear-solvers": ("matsim", "skills/core-numerical/nonlinear-solvers", [], False),
    "numerical-integration": ("matsim", "skills/core-numerical/numerical-integration", [], False),
    "numerical-stability": ("matsim", "skills/core-numerical/numerical-stability", [], False),
    "time-stepping": ("matsim", "skills/core-numerical/time-stepping", [], False),
    "benchmark-and-mms-planner": ("matsim", "skills/verification-validation/benchmark-and-mms-planner", [], False),
    # -- tondevrel/scientific-agent-skills (confirmed non-duplicate w/ K-Dense) --
    "scikit-image": ("tondevrel", "skills/scikit-image", ["scikit-image"], False),
    "scipy": ("tondevrel", "skills/scipy", ["scipy"], False),
    "dowhy": ("tondevrel", "skills/dowhy", ["dowhy"], False),  # upstream ships SKILL.MD (uppercase); see _resolve_source
    "xgboost-lightgbm": ("tondevrel", "skills/xgboost-lightgbm", ["xgboost", "lightgbm"], False),
    "lifelines": ("tondevrel", "skills/lifelines", ["lifelines"], False),
    # -- MP-AI-20/mechanical-engineering-skills (curated subset) --
    "control-systems": ("mecheng", "skills/control-systems", [], False),
    "state-space-control": ("mecheng", "skills/state-space-control", [], False),
    "pid-controller": ("mecheng", "skills/pid-controller", [], False),
    "lqr-control": ("mecheng", "skills/lqr-control", [], False),
    "kalman-filter": ("mecheng", "skills/kalman-filter", [], False),
    "weibull-analysis": ("mecheng", "skills/weibull-analysis", [], False),
    "reliability-engineering": ("mecheng", "skills/reliability-engineering", [], False),
    "accelerated-life-testing": ("mecheng", "skills/accelerated-life-testing", [], False),
    "system-reliability": ("mecheng", "skills/system-reliability", [], False),
    "structural-analysis": ("mecheng", "skills/structural-analysis", [], False),
    "truss-analysis": ("mecheng", "skills/truss-analysis", [], False),
    "beam-solver": ("mecheng", "skills/beam-solver", [], False),
    "fea-fundamentals": ("mecheng", "skills/fea-fundamentals", [], False),
    "column-buckling": ("mecheng", "skills/column-buckling", [], False),
    "topology-optimization": ("mecheng", "skills/topology-optimization", [], False),
    # -- Z-Richard/Atmos-sci-skills (all four) --
    "metpy": ("atmos", "skills/metpy", ["metpy"], False),
    "map": ("atmos", "skills/map", ["cartopy"], False),
    "regrid": ("atmos", "skills/regrid", [], False),
    "animate": ("atmos", "skills/animate", [], False),
    # -- jaechang-hits/SciAgent-Skills (one skill; rest overlaps K-Dense) --
    "neurokit2": ("sciagent", "skills/scientific-computing/neurokit2", ["neurokit2"], False),
    # -- Learning-Bayesian-Statistics/baygent-skills (skip bayesian-workflow: redundant w/ pymc) --
    "causal-inference": ("baygent", "causal-inference", [], False),
    "amortized-workflow": ("baygent", "amortized-workflow", [], False),
    # -- HughYau/neuroforge-skills (skip pynibs: narrow TMS use case) --
    "mne-python": ("neuroforge", "skills/mne-python", ["mne"], False),
    "brian2": ("neuroforge", "skills/brian2", ["brian2"], False),
    "nilearn": ("neuroforge", "skills/nilearn", ["nilearn"], False),
    "spikeinterface": ("neuroforge", "skills/spikeinterface", ["spikeinterface"], False),
}

ASSETS_DIR = Path(__file__).resolve().parent / "skill-assets" / "lieflat-charts"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def _clone_or_refresh(url: str, dest: Path, sparse_paths: list[str] | None = None) -> None:
    """Clone (or refresh) a source repo, scoped to only the paths this
    script actually imports from it, via ``git sparse-checkout``.

    Two independent reasons this matters, not just speed: some of these
    repos bundle 100+ skills when only a handful are wanted (ClawBio: 6
    of 96; mechanical-engineering-skills: 15 of 647) -- checking out the
    rest is pure waste. Worse, a full checkout can outright FAIL on
    Windows: ClawBio ships test fixtures nested deep enough
    (``skills/locuscompare-region-render/tests/fixtures/golden/...``) to
    exceed the 260-char MAX_PATH, and that skill isn't even one this
    script imports -- an unrelated file blocking the whole clone.
    ``sparse_paths=None`` (lieflat-charts, small and taken whole) does a
    normal full clone.
    """
    if (dest / ".git").is_dir():
        print(f"  refreshing {dest.name} ...")
        try:
            _run(["git", "-C", str(dest), "pull", "--ff-only"], capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"  (pull failed, using the existing clone as-is: {e})")
        return
    print(f"  cloning {url} -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if sparse_paths:
            _run(
                ["git", "clone", "--depth", "1", "--filter=blob:none",
                 "--sparse", url, str(dest)],
                capture_output=True, text=True,
            )
            _run(
                ["git", "-C", str(dest), "sparse-checkout", "set", *sparse_paths],
                capture_output=True, text=True,
            )
        else:
            _run(["git", "clone", "--depth", "1", url, str(dest)], capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e}")
        if e.stderr:
            print(f"  {e.stderr.strip()[-500:]}")


def _prepare_lieflat(clone_dir: Path) -> None:
    """Layer FI's static-render addition onto the fresh clone."""
    scripts_dir = clone_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    shutil.copy2(ASSETS_DIR / "render_static.mjs", scripts_dir / "render_static.mjs")
    skill_md = clone_dir / "SKILL.md"
    addendum = (ASSETS_DIR / "ADDENDUM.md").read_text(encoding="utf-8")
    body = skill_md.read_text(encoding="utf-8")
    marker = "## FI addendum (not part of the upstream skill)"
    if marker not in body:
        skill_md.write_text(body.rstrip() + "\n\n---\n\n" + addendum, encoding="utf-8")


def _resolve_source(skill_dir: Path) -> Path:
    """Point at a skill's SKILL.md file directly rather than its directory.

    ``core.skills.importer._find_skill_md`` looks for a child literally
    named ``SKILL.md`` when given a directory -- exact case, because it's
    a plain path join, not a filesystem-level case-insensitive lookup.
    That's invisible on Windows (NTFS ignores case) and a real failure on
    Linux, where at least one upstream skill (tondevrel's ``dowhy``) ships
    ``SKILL.MD`` uppercase. Passing the FILE directly instead of the
    directory sidesteps the join entirely -- ``_find_skill_md`` accepts
    any path whose name case-insensitively ends in ``.md``.
    """
    for name in ("SKILL.md", "SKILL.MD", "skill.md"):
        candidate = skill_dir / name
        if candidate.is_file():
            return candidate
    return skill_dir  # let the importer raise its own FileNotFoundError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cache-dir", default=str(REPO_ROOT / ".skill-sources"),
        help="Where source repos are cloned (default: .skill-sources/ next to this script; "
             "gitignored, not shipped).",
    )
    ap.add_argument("--skip", default="", help="Comma-separated skill names to skip.")
    ap.add_argument(
        "--pip-install", action="store_true",
        help="Also pip-install the underlying packages into this interpreter. Off by default.",
    )
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    import launch  # noqa: E402  (path set up above)

    # Some upstream skills carry non-ASCII frontmatter (lieflat-charts' own
    # description is Chinese) that _import_skill() prints straight through.
    # ``python launch.py ...`` already guards against a Windows cp1252
    # console choking on that (see launch._force_utf8_streams's docstring);
    # calling _import_skill() directly, as this script does, bypasses
    # main() and skips that guard, so it has to be called here too.
    launch._force_utf8_streams()

    # Every relative path this script actually imports, grouped by repo --
    # the sparse-checkout scope. A repo with no entry here (or an entry
    # containing "") is taken whole (lieflat-charts: small, and the
    # skill IS the repo root).
    sparse_by_repo: dict[str, list[str]] = {}
    for repo_key, rel, *_ in SKILLS.values():
        if not rel:
            sparse_by_repo[repo_key] = []  # sentinel: take the whole repo
            continue
        sparse_by_repo.setdefault(repo_key, [])
        if rel not in sparse_by_repo[repo_key]:
            sparse_by_repo[repo_key].append(rel)

    needed_repo_keys = {repo_key for repo_key, *_ in SKILLS.values()}
    print(f"Fetching {len(needed_repo_keys)} source repos ...")
    repo_dirs: dict[str, Path] = {}
    for key, url in REPOS.items():
        # Keyed on OUR internal repo key, not the URL's basename: two
        # different owners here both named their repo
        # "scientific-agent-skills" (K-Dense-AI's and tondevrel's) --
        # naming the cache dir from the URL alone collided them onto the
        # same local folder, so the second clone silently "refreshed"
        # (git pull) into what was actually the first repo's checkout.
        dest = cache_dir / key
        repo_dirs[key] = dest
        paths = sparse_by_repo.get(key, [])
        _clone_or_refresh(url, dest, sparse_paths=(paths or None))
    _prepare_lieflat(repo_dirs["lieflat"])
    print()

    results: list[tuple[str, bool, bool, list[str]]] = []  # name, imported, needs_despite, pip_pkgs
    for name, (repo_key, rel, pip_pkgs, needs_despite) in SKILLS.items():
        if name in skip:
            print(f"-- {name}: skipped")
            continue
        repo_dir = repo_dirs[repo_key]
        skill_dir = repo_dir / rel if rel else repo_dir
        source = _resolve_source(skill_dir)
        print(f"-- {name}")
        # Persist the curated package list into the skill's provenance so the
        # mapping outlives this script -- `--approve-all-skills --pip-install`
        # reads it to fix quarantined skills without re-deriving it.
        rc = launch._import_skill(
            str(source), name, domains="", pip_requires=list(pip_pkgs),
        )
        imported = rc == 0
        results.append((name, imported, needs_despite, pip_pkgs))
        print()

    all_pip: list[str] = []
    for _, _, _, pkgs in results:
        for p in pkgs:
            if p not in all_pip:
                all_pip.append(p)

    if args.pip_install and all_pip:
        print(f"Installing underlying packages: {' '.join(all_pip)}")
        subprocess.run([sys.executable, "-m", "pip", "install", *all_pip])
        print()
    elif all_pip:
        print("Underlying packages not installed (pass --pip-install, or run yourself):")
        print(f"  {sys.executable} -m pip install {' '.join(all_pip)}")
        print()

    # Deliberately a placeholder rather than `git config user.name`. The ledger
    # records who reviewed a skill, and whoever runs this import is often not
    # whoever ends up reviewing what it pulled in — on a shared clone they are
    # frequently different people. Printing a real name makes the wrong
    # attribution the copy-pasteable default, which is the thing the
    # no-anonymous-approver rule exists to prevent. `<you>` is also a redirect
    # in every common shell, so the line cannot be pasted without being edited.
    approve_as = "<you>"

    imported_count = sum(1 for _, ok, _, _ in results if ok)
    any_despite = any(nd for _, ok, nd, _ in results if ok)
    print(f"Imported {imported_count}/{len(results)}. Nothing above was approved.")
    print("Replace <you> with the name of whoever actually reviewed the skills.
")
    print("Review them, then approve in one go (installs missing packages first,")
    print("re-tests, and still refuses anything whose self-test fails):")
    despite_flag = " --despite-findings" if any_despite else ""
    print(
        f"  python launch.py --approve-all-skills --approve-as {approve_as} "
        f"--pip-install{despite_flag}"
    )
    if any_despite:
        print(
            "
  (--despite-findings is included because some of these carry "
            "high-severity
   scan findings. Read them first: "
            "python launch.py --scan-skill <name>)"
        )
    print("
Or one at a time:")
    for name, imported, needs_despite, _ in results:
        if not imported:
            continue
        flag = " --despite-findings" if needs_despite else ""
        print(f"  python launch.py --approve-skill {name} --approve-as {approve_as}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
