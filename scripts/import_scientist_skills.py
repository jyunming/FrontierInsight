"""Reproduce this project's curated scientist-workflow skill set from scratch.

FI ships no skills — the discovery root is user state
(``~/.frontier-insight/skills``, or ``FI_SKILLS_DIR``), not repository
content, so a fresh clone of this repo starts with an empty skill library
(see ``core/skills/registry.py``). The thirteen skills below were sourced
and reviewed in one working session; this script reproduces the *sourcing*
step so nobody has to re-derive the source repo, the folder names, and the
command sequence by hand. It does NOT approve anything for you — approval
binds to a person's judgement (``--approve-skill --approve-as WHO``) and is
deliberately not a thing a script gets to do.

Sources:
  * Twelve skills from K-Dense-AI/scientific-agent-skills (GitHub) — an
    Agent Skills-format catalogue already behind three skills already
    trusted in this library (scikit-learn, get-available-resources,
    iso-standards-readiness, what-if-oracle), so format and quality are a
    known quantity.
  * lieflat-charts, from larashero3-dotcom/lieflat-charts, with a small
    FI-authored addition bundled alongside this script
    (``scripts/skill-assets/lieflat-charts/``): a static-render step, since
    FI's paper_pdf/poster/slides outputs can't embed a live JS chart.

What this does, per skill:
  1. Clone (or refresh an already-cloned) source repo into ``--cache-dir``.
  2. Import via ``core.skills.importer`` — lands UNTESTED; a self-test is
     auto-generated if the source didn't ship one, same as any import.
  3. Run the static scan (``core.skills.scan``) and report the summary.
  4. Print the exact approve command. Never runs it.

Usage:
    python scripts/import_scientist_skills.py
    python scripts/import_scientist_skills.py --cache-dir /path/to/cache
    python scripts/import_scientist_skills.py --skip pymc,rdkit
    python scripts/import_scientist_skills.py --pip-install

``--pip-install`` additionally installs the underlying Python packages
(pymc, astropy, rdkit, ...) into *this* interpreter's environment — a real,
visible side effect on what may be a shared install. Off by default; either
way, the exact pip line is printed so you can run it yourself.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

KDENSE_REPO = "https://github.com/K-Dense-AI/scientific-agent-skills.git"
LIEFLAT_REPO = "https://github.com/larashero3-dotcom/lieflat-charts.git"

# name -> (repo key, relative path inside that clone, pip packages, needs --despite-findings)
SKILLS: dict[str, tuple[str, str, list[str], bool]] = {
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
    "lieflat-charts": ("lieflat", "", [], False),
}

ASSETS_DIR = Path(__file__).resolve().parent / "skill-assets" / "lieflat-charts"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def _clone_or_refresh(url: str, dest: Path) -> None:
    if (dest / ".git").is_dir():
        print(f"  refreshing {dest.name} ...")
        try:
            _run(["git", "-C", str(dest), "pull", "--ff-only"], capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"  (pull failed, using the existing clone as-is: {e})")
        return
    print(f"  cloning {url} -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", url, str(dest)], capture_output=True, text=True)


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

    print("Fetching source repos ...")
    kdense_dir = cache_dir / "scientific-agent-skills"
    lieflat_dir = cache_dir / "lieflat-charts"
    _clone_or_refresh(KDENSE_REPO, kdense_dir)
    _clone_or_refresh(LIEFLAT_REPO, lieflat_dir)
    _prepare_lieflat(lieflat_dir)
    print()

    results: list[tuple[str, bool, bool, list[str]]] = []  # name, imported, needs_despite, pip_pkgs
    for name, (repo_key, rel, pip_pkgs, needs_despite) in SKILLS.items():
        if name in skip:
            print(f"-- {name}: skipped")
            continue
        source = (kdense_dir / rel) if repo_key == "kdense" else lieflat_dir
        print(f"-- {name}")
        rc = launch._import_skill(str(source), name, domains="")
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

    try:
        approve_as = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True, check=True,
        ).stdout.strip() or "<you>"
    except (subprocess.CalledProcessError, FileNotFoundError):
        approve_as = "<you>"

    print("Nothing above was approved. Review each, then:")
    for name, imported, needs_despite, _ in results:
        if not imported:
            continue
        flag = " --despite-findings" if needs_despite else ""
        print(f"  python launch.py --approve-skill {name} --approve-as {approve_as}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
