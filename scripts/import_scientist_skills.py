"""Reproduce this project's curated scientist-workflow skill set from scratch.

FI ships no skills — the discovery root is user state
(``~/.frontier-insight/skills``, or ``FI_SKILLS_DIR``), not repository
content, so a fresh clone of this repo starts with an empty skill library
(see ``core/skills/registry.py``). The skills below were sourced and
reviewed across two working sessions; this script reproduces the
*sourcing* step so nobody has to re-derive ten source repos, seventy-odd
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
    python scripts/import_scientist_skills.py --offline        # use the clones already in the cache

Fetching cannot hang: git is never allowed to ask for anything, a transfer that stalls is
dropped, and a clone or pull still running after ``--git-timeout`` seconds (default 300) is
stopped with everything it started. A repository that cannot be fetched is named at the end
with the reason, its skills are skipped, and the rest are imported; the exit code is then 1.
The default cache is ``.skill-sources/`` next to this script, or ``~/.frontier-insight/skill-sources``
when that would sit inside a OneDrive folder (a sync client stalls git or locks its files).

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
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPOS: dict[str, str] = {
    "kdense": "https://github.com/K-Dense-AI/scientific-agent-skills.git",
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

# name -> (repo key, relative path inside that clone, needs --despite-findings). The pip packages each needs are
# in core/skills/known_requirements.py (PRESET_PIP_REQUIRES), which a quest reads too.
SKILLS: dict[str, tuple[str, str, bool]] = {
    # -- K-Dense-AI/scientific-agent-skills --
    "pymc": ("kdense", "skills/pymc", False),
    "astropy": ("kdense", "skills/astropy", False),
    "rdkit": ("kdense", "skills/rdkit", False),
    "biopython": ("kdense", "skills/biopython", False),
    "pymatgen": ("kdense", "skills/pymatgen", False),
    "scikit-bio": ("kdense", "skills/scikit-bio", False),
    "geopandas": ("kdense", "skills/geopandas", False),
    "cobrapy": ("kdense", "skills/cobrapy", False),
    "qutip": ("kdense", "skills/qutip", False),
    "scientific-critical-thinking": ("kdense", "skills/scientific-critical-thinking", False),
    "ontology-term-resolution": ("kdense", "skills/ontology-term-resolution", True),
    "molecular-dynamics": ("kdense", "skills/molecular-dynamics", False),
    # Python figures for the experiment's own plots (they replaced lieflat-charts, a Node.js HTML-chart tool a
    # quest's Python experiment could not import): publication styles and palettes, plus the two plotting libraries.
    "matplotlib": ("kdense", "skills/matplotlib", False),
    "seaborn": ("kdense", "skills/seaborn", False),
    "scientific-visualization": ("kdense", "skills/scientific-visualization", False),
    # -- ClawBio/ClawBio (curated subset -- see module docstring) --
    "analyze-fasta": ("clawbio", "skills/analyze-fasta", False),
    "genome-compare": ("clawbio", "skills/genome-compare", False),
    "variant-annotation": ("clawbio", "skills/variant-annotation", False),
    "vcf-annotator": ("clawbio", "skills/vcf-annotator", False),
    "phylogenetics-builder": ("clawbio", "skills/phylogenetics-builder", False),
    "methylation-clock": ("clawbio", "skills/methylation-clock", False),
    # -- SteadfastAsArt/geoscience-skills (flat at repo root, not under skills/) --
    "obspy": ("geoscience", "obspy", False),
    "lasio": ("geoscience", "lasio", False),
    "welly": ("geoscience", "welly", False),
    "gempy": ("geoscience", "gempy", False),
    # simpeg's bundled scripts import discretize directly (mesh generation),
    # not just simpeg's own re-exports -- pip-installing simpeg alone leaves
    # `from discretize import TensorMesh` unresolved.
    "simpeg": ("geoscience", "simpeg", False),
    "harmonica": ("geoscience", "harmonica", False),
    # Pinned: landlab 2.11.0's own code uses Python 3.12+ generic-function
    # syntax (`def f[T](...)`) despite the package claiming
    # `requires_python: >=3.11` on PyPI -- a real upstream metadata bug,
    # verified live (2.11.0 fails a bare `import landlab` with SyntaxError
    # under 3.11; 2.10.1 imports cleanly). Re-check this pin once landlab
    # either fixes the classifier or 3.11 support is genuinely dropped.
    "landlab": ("geoscience", "landlab", False),
    "pastas": ("geoscience", "pastas", False),
    "segyio": ("geoscience", "segyio", False),
    "xarray": ("geoscience", "xarray", False),
    # -- HeshamFS/materials-simulation-skills --
    "convergence-study": ("matsim", "skills/core-numerical/convergence-study", False),
    "differentiation-schemes": ("matsim", "skills/core-numerical/differentiation-schemes", False),
    "linear-solvers": ("matsim", "skills/core-numerical/linear-solvers", False),
    "mesh-generation": ("matsim", "skills/core-numerical/mesh-generation", False),
    "nonlinear-solvers": ("matsim", "skills/core-numerical/nonlinear-solvers", False),
    "numerical-integration": ("matsim", "skills/core-numerical/numerical-integration", False),
    "numerical-stability": ("matsim", "skills/core-numerical/numerical-stability", False),
    "time-stepping": ("matsim", "skills/core-numerical/time-stepping", False),
    "benchmark-and-mms-planner": ("matsim", "skills/verification-validation/benchmark-and-mms-planner", False),
    # -- tondevrel/scientific-agent-skills (confirmed non-duplicate w/ K-Dense) --
    "scikit-image": ("tondevrel", "skills/scikit-image", False),
    "scipy": ("tondevrel", "skills/scipy", False),
    "dowhy": ("tondevrel", "skills/dowhy", False),  # upstream ships SKILL.MD (uppercase); see _resolve_source
    "xgboost-lightgbm": ("tondevrel", "skills/xgboost-lightgbm", False),
    "lifelines": ("tondevrel", "skills/lifelines", False),
    # -- MP-AI-20/mechanical-engineering-skills (curated subset) --
    "control-systems": ("mecheng", "skills/control-systems", False),
    "state-space-control": ("mecheng", "skills/state-space-control", False),
    "pid-controller": ("mecheng", "skills/pid-controller", False),
    "lqr-control": ("mecheng", "skills/lqr-control", False),
    "kalman-filter": ("mecheng", "skills/kalman-filter", False),
    "weibull-analysis": ("mecheng", "skills/weibull-analysis", False),
    "reliability-engineering": ("mecheng", "skills/reliability-engineering", False),
    "accelerated-life-testing": ("mecheng", "skills/accelerated-life-testing", False),
    "system-reliability": ("mecheng", "skills/system-reliability", False),
    "structural-analysis": ("mecheng", "skills/structural-analysis", False),
    "truss-analysis": ("mecheng", "skills/truss-analysis", False),
    "beam-solver": ("mecheng", "skills/beam-solver", False),
    "fea-fundamentals": ("mecheng", "skills/fea-fundamentals", False),
    "column-buckling": ("mecheng", "skills/column-buckling", False),
    "topology-optimization": ("mecheng", "skills/topology-optimization", False),
    # -- Z-Richard/Atmos-sci-skills (all four) --
    "metpy": ("atmos", "skills/metpy", False),
    "map": ("atmos", "skills/map", False),
    "regrid": ("atmos", "skills/regrid", False),
    "animate": ("atmos", "skills/animate", False),
    # -- jaechang-hits/SciAgent-Skills (one skill; rest overlaps K-Dense) --
    "neurokit2": ("sciagent", "skills/scientific-computing/neurokit2", False),
    # -- Learning-Bayesian-Statistics/baygent-skills (skip bayesian-workflow: redundant w/ pymc) --
    "causal-inference": ("baygent", "causal-inference", False),
    "amortized-workflow": ("baygent", "amortized-workflow", False),
    # -- HughYau/neuroforge-skills (skip pynibs: narrow TMS use case) --
    "mne-python": ("neuroforge", "skills/mne-python", False),
    "brian2": ("neuroforge", "skills/brian2", False),
    "nilearn": ("neuroforge", "skills/nilearn", False),
    "spikeinterface": ("neuroforge", "skills/spikeinterface", False),
}

# One clone or pull gets this long. A sparse, shallow clone of a large repository takes a few
# minutes on a slow company link; a stalled connection would take for ever.
GIT_TIMEOUT_S = 300
# ``core.longpaths`` lifts Windows' 260-character limit for what git writes (a checkout under a
# deep folder such as OneDrive's is otherwise one nested test fixture away from failing).
GIT = ["git", "-c", "core.longpaths=true"]


class GitResult:
    def __init__(self, returncode: int, output: str, timed_out: bool = False) -> None:
        self.returncode = returncode
        self.output = output
        self.timed_out = timed_out

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def reason(self) -> str:
        if self.timed_out:
            return "timed out"
        lines = [line.strip() for line in self.output.strip().splitlines() if line.strip()]
        # git names what went wrong on a "fatal:" or "error:" line, and may add advice after it.
        for line in reversed(lines):
            if line.lower().startswith(("fatal:", "error:")):
                return line[:200]
        return (lines[-1] if lines else f"exit code {self.returncode}")[:200]


def _git_env() -> dict[str, str]:
    """git must never wait for a person, and must give up on a transfer that has stalled."""
    env = dict(os.environ)
    env.update({
        # A credential prompt, or a credential-manager window behind a console whose output is
        # being captured, is a silent hang.
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        # A proxy that accepts the connection and then says nothing.
        "GIT_HTTP_LOW_SPEED_LIMIT": "1000",
        "GIT_HTTP_LOW_SPEED_TIME": "60",
    })
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    """Stop ``proc`` and everything it started. Killing git alone leaves the helper it started
    for the network (git-remote-https) running."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, stdin=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _git(argv: list[str], *, timeout: float) -> GitResult:
    """Run one git command that cannot hang.

    Its output goes to a file, not a pipe: a pipe stays open for as long as any helper git started
    is alive, so a run that had to be stopped could not even be read (``subprocess.run`` with
    ``capture_output`` sat in ``communicate`` for ever, which is what a stalled ``git pull`` looked
    like). It never asks for anything, and a command still running after ``timeout`` seconds is
    stopped with everything it started."""
    kwargs: dict = {} if os.name == "nt" else {"start_new_session": True}
    timed_out = False
    try:
        with tempfile.TemporaryFile() as out:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                env=_git_env(), **kwargs,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_tree(proc)
                proc.wait()
            out.seek(0)
            text = out.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return GitResult(127, f"{argv[0]} was not found on PATH: install git, or put it on PATH")
    return GitResult(-1 if timed_out else proc.returncode, text, timed_out)


def _remove_tree(path: Path) -> None:
    """Delete a folder, including the read-only files git keeps under ``.git`` on Windows."""
    def writable(func, name, _exc) -> None:
        os.chmod(name, stat.S_IWRITE)
        func(name)

    shutil.rmtree(path, onerror=writable)


def _default_cache_dir() -> Path:
    """``.skill-sources/`` next to the repository, unless that sits inside a OneDrive folder: a sync
    client that watches a git checkout stalls git or locks its files, and the folder's own path is
    already most of Windows' 260 characters. Then ``~/.frontier-insight/skill-sources``, which
    nothing syncs."""
    inside = REPO_ROOT / ".skill-sources"
    if any(part.lower().startswith("onedrive") for part in inside.resolve().parts):
        return Path.home() / ".frontier-insight" / "skill-sources"
    return inside


def _clone_or_refresh(
    url: str, dest: Path, sparse_paths: list[str] | None = None, *,
    timeout: float = GIT_TIMEOUT_S, offline: bool = False,
) -> tuple[bool, str]:
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
    ``sparse_paths=None`` (a skill that is its whole repository) does a
    normal full clone.

    Returns ``(usable, why_not)``: whether there is a clone to import from afterwards. A refresh
    that fails keeps the clone that is already there. A first clone that fails is removed, so the
    next run clones again instead of "refreshing" a folder with nothing checked out in it.
    """
    have = (dest / ".git").is_dir()
    if offline:
        if have:
            print(f"  {dest.name}: offline, using the clone already here", flush=True)
            return True, ""
        return False, "not cloned yet, and --offline was given"
    if have:
        print(f"  refreshing {dest.name} ...", flush=True)
        pulled = _git([*GIT, "-C", str(dest), "pull", "--ff-only"], timeout=timeout)
        if not pulled.ok:
            print(f"  ({dest.name}: pull failed, {pulled.reason()}; using the existing clone as-is)", flush=True)
        if sparse_paths:
            # A first run stopped between the clone and this step left a clone with nothing checked out.
            scoped = _git([*GIT, "-C", str(dest), "sparse-checkout", "set", *sparse_paths], timeout=timeout)
            if not scoped.ok and not all((dest / rel).exists() for rel in sparse_paths):
                return False, f"its skill folders could not be checked out ({scoped.reason()})"
        return True, ""
    print(f"  cloning {url} -> {dest} ...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sparse_paths:
        cloned = _git(
            [*GIT, "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)],
            timeout=timeout,
        )
        if cloned.ok:
            cloned = _git([*GIT, "-C", str(dest), "sparse-checkout", "set", *sparse_paths], timeout=timeout)
    else:
        cloned = _git([*GIT, "clone", "--depth", "1", url, str(dest)], timeout=timeout)
    if not cloned.ok:
        if dest.exists():
            _remove_tree(dest)
        return False, cloned.reason()
    return True, ""


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
        "--cache-dir", default=None,
        help="Where source repos are cloned (default: .skill-sources/ next to this script, "
             "gitignored, not shipped; ~/.frontier-insight/skill-sources when that would be "
             "inside a OneDrive folder).",
    )
    ap.add_argument(
        "--git-timeout", type=int, default=GIT_TIMEOUT_S, metavar="SECONDS",
        help=f"Give up on one clone or pull after this long (default {GIT_TIMEOUT_S}).",
    )
    ap.add_argument(
        "--offline", action="store_true",
        help="Do not touch the network: import from the clones already in the cache "
             "(clone a repository there yourself if git cannot reach it from this machine).",
    )
    ap.add_argument("--skip", default="", help="Comma-separated skill names to skip.")
    ap.add_argument(
        "--pip-install", action="store_true",
        help="Also pip-install the underlying packages into this interpreter. Off by default.",
    )
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else _default_cache_dir()

    import launch  # noqa: E402  (path set up above)
    from core.skills import known_requirements  # noqa: E402

    # Some upstream skills carry non-ASCII frontmatter (a description in
    # Chinese, say) that _import_skill() prints straight through.
    # ``python launch.py ...`` already guards against a Windows cp1252
    # console choking on that (see launch._force_utf8_streams's docstring);
    # calling _import_skill() directly, as this script does, bypasses
    # main() and skips that guard, so it has to be called here too.
    launch._force_utf8_streams()

    # Every relative path this script actually imports, grouped by repo --
    # the sparse-checkout scope. A repo with no entry here (or an entry
    # containing "") is taken whole (a skill that IS its repo root).
    sparse_by_repo: dict[str, list[str]] = {}
    for repo_key, rel, *_ in SKILLS.values():
        if not rel:
            sparse_by_repo[repo_key] = []  # sentinel: take the whole repo
            continue
        sparse_by_repo.setdefault(repo_key, [])
        if rel not in sparse_by_repo[repo_key]:
            sparse_by_repo[repo_key].append(rel)

    needed_repo_keys = {repo_key for repo_key, *_ in SKILLS.values()}
    print(f"Source repos are kept in {cache_dir}")
    print(
        f"{'Using the clones already there' if args.offline else 'Fetching'} "
        f"{len(needed_repo_keys)} source repos ...",
        flush=True,
    )
    repo_dirs: dict[str, Path] = {}
    unfetched: dict[str, str] = {}  # repo key -> why there is nothing to import from
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
        usable, why = _clone_or_refresh(
            url, dest, sparse_paths=(paths or None), timeout=args.git_timeout, offline=args.offline,
        )
        if not usable:
            unfetched[key] = why
            print(f"  {key}: NOT FETCHED, {why}", flush=True)
    if unfetched:
        print()
        print(f"Could not fetch {len(unfetched)} of {len(REPOS)} source repos: {', '.join(unfetched)}.")
        print("Their skills are skipped; everything else is imported. On a company network the usual causes are")
        print("a proxy git does not know about (git config --global http.proxy http://host:port), a VPN that")
        print("is off, an SSL-inspection certificate (git config --global http.sslBackend schannel), or a")
        print("synced folder locking git's files (pass --cache-dir a folder nothing syncs). Or clone them")
        print("yourself, from a machine or a shell that can reach GitHub, and run again with --offline:")
        for key in unfetched:
            print(f"  git -c core.longpaths=true clone --depth 1 {REPOS[key]} {cache_dir / key}")
    print()

    results: list[tuple[str, bool, bool, list[str]]] = []  # name, imported, needs_despite, pip_pkgs
    not_attempted: list[str] = []
    for name, (repo_key, rel, needs_despite) in SKILLS.items():
        if name in skip:
            print(f"-- {name}: skipped")
            continue
        if repo_key in unfetched:
            print(f"-- {name}: skipped (its source repo {repo_key} was not fetched)")
            not_attempted.append(name)
            continue
        repo_dir = repo_dirs[repo_key]
        skill_dir = repo_dir / rel if rel else repo_dir
        source = _resolve_source(skill_dir)
        print(f"-- {name}")
        # Persist the curated package list into the skill's provenance so the
        # mapping outlives this script -- `--approve-all-skills --pip-install`
        # reads it to fix quarantined skills without re-deriving it.
        # The package list comes from core/skills/known_requirements.py, which a quest also reads for a skill
        # imported before provenance recorded one.
        pip_pkgs = known_requirements.PRESET_PIP_REQUIRES.get(name, [])
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
        _installed, failed = launch._pip_install(all_pip)
        if failed:
            print(f"Could not install: {', '.join(failed)} (the reason is above). The skills that need them fail")
            print("their self-test until they are installed; the rest are unaffected.")
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
    if not_attempted:
        print(f"Not attempted, because their source repo could not be fetched: {len(not_attempted)} skills.")
    print("Replace <you> with the name of whoever actually reviewed the skills.\n")
    print("Review them, then approve in one go (installs missing packages first,")
    print("re-tests, and still refuses anything whose self-test fails):")
    despite_flag = " --despite-findings" if any_despite else ""
    print(
        f"  python launch.py --approve-all-skills --approve-as {approve_as} "
        f"--pip-install{despite_flag}"
    )
    if any_despite:
        print(
            "\n  (--despite-findings is included because some of these carry "
            "high-severity\n   scan findings. Read them first: "
            "python launch.py --scan-skill <name>)"
        )
    print("\nOr one at a time:")
    for name, imported, needs_despite, _ in results:
        if not imported:
            continue
        flag = " --despite-findings" if needs_despite else ""
        print(f"  python launch.py --approve-skill {name} --approve-as {approve_as}{flag}")
    return 1 if unfetched else 0


if __name__ == "__main__":
    sys.exit(main())
