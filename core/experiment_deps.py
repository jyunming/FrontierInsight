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

Package names live here too, once: the pip name a package is installed by is not always the name it is imported by
(``scikit-image`` is ``import skimage``). The requested list is what the model said it would use, not what the scripts
import, so it is checked against the scripts (:func:`plan_installs`): a listed package no script imports is not
installed, and a well-known package a script imports but the list lacks is. The test import after installing
(:func:`warmup_modules`) imports only names that are known, never a guess.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# The distribution name at the start of a requirement ("numpy>=1.26", "scikit-learn[all]", "pkg ; python_version>..").
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize(name: str) -> str:
    """PEP 503 form: lower case, runs of ``-``, ``_`` and ``.`` as one ``-`` (so ``lieflat_charts`` is ``lieflat-charts``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(dep: str) -> str:
    """The distribution name of a requirement string, or ``""`` when it has none (a URL, a path). A direct reference
    (``scikit-image @ git+https://...``) has the name before the ``@``."""
    dep = str(dep)
    head, at, rest = dep.partition("@")
    if at and re.match(r"\s*([A-Za-z][\w+.-]*://|file:)", rest) and _NAME_RE.fullmatch(head.split("[")[0].strip() or "-"):
        return head.split("[")[0].strip()
    m = _NAME_RE.match(dep)
    return m.group(1) if m and "/" not in dep and "\\" not in dep else ""


def split_deps(
    deps: Iterable[str], *, local_modules: Iterable[str],
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
    """The packages the selected skills need: their recorded ``pip_requires``, or for a preset skill imported before
    provenance recorded one, the preset table's list (core/skills/known_requirements.py); each pinned where a bare name
    installs a broken version."""
    from core.skills import known_requirements

    out: list[str] = []
    for s in skills:
        out.extend(known_requirements.pip_requires(s))
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


# ---------------------------------------------------------------------------------------------------------------------
# Package names: the pip name a package is installed by, and the name it is imported by.
# ---------------------------------------------------------------------------------------------------------------------

# Import name -> the pip package that provides it, only where the two differ. Well-known packages only: an import name
# is never sent to pip on a guess (PyPI's ``serial`` is not pyserial), so a name missing here is left to the run.
IMPORT_TO_PIP: dict[str, str] = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "dateutil": "python-dateutil",
    "fitz": "PyMuPDF",
    "google.protobuf": "protobuf",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "serial": "pyserial",
    "Crypto": "pycryptodome",
    "jwt": "PyJWT",
    "dotenv": "python-dotenv",
    "Bio": "biopython",
    "pywt": "PyWavelets",
    "skopt": "scikit-optimize",
    "skfuzzy": "scikit-fuzzy",
    "umap": "umap-learn",
    "OpenGL": "PyOpenGL",
    "zmq": "pyzmq",
    "osgeo": "GDAL",
    "MySQLdb": "mysqlclient",
    "community": "python-louvain",
    "attr": "attrs",
    "ruamel.yaml": "ruamel.yaml",  # the same name, but under a namespace: the dotted name is the one to read
}

# Pip package (PEP 503 form) -> the names it is imported by, where they differ from the pip name. The reverse of
# ``IMPORT_TO_PIP``, plus the other packages that provide the same module (the opencv builds, psycopg2-binary, ...).
_PIP_TO_IMPORTS: dict[str, list[str]] = {}
for _module, _pip in IMPORT_TO_PIP.items():
    _PIP_TO_IMPORTS.setdefault(normalize(_pip), []).append(_module)
for _pip, _modules in {
    "opencv-python-headless": ["cv2"],
    "opencv-contrib-python": ["cv2"],
    "opencv-contrib-python-headless": ["cv2"],
    "pymupdf": ["fitz", "pymupdf"],
    "attrs": ["attr", "attrs"],
    "psycopg2-binary": ["psycopg2"],
    "pycrypto": ["Crypto"],
    "tensorflow-cpu": ["tensorflow"],
    "faiss-cpu": ["faiss"],
    "faiss-gpu": ["faiss"],
    "ruamel-yaml": ["ruamel.yaml"],
    "pyqt5": ["PyQt5"],
    "pyqt6": ["PyQt6"],
    "pyside2": ["PySide2"],
    "pyside6": ["PySide6"],
}.items():
    _PIP_TO_IMPORTS[_pip] = _modules
del _module, _pip, _modules

# Packages other libraries load by name while the script runs, so a script can need one it never imports
# (``pandas.read_excel`` needs openpyxl, ``fig.write_image`` needs kaleido, ``to_parquet`` needs pyarrow, an
# xarray engine needs netCDF4, a matplotlib window needs a Qt binding). A listed one is always installed.
_LOADED_BY_OTHERS = frozenset(normalize(n) for n in (
    "openpyxl", "xlrd", "xlsxwriter", "odfpy", "pyxlsb", "pyarrow", "fastparquet", "tables", "kaleido", "tabulate",
    "jinja2", "lxml", "html5lib", "numexpr", "bottleneck", "netCDF4", "h5netcdf", "cftime", "zarr", "dask", "scipy",
    "statsmodels", "PyQt5", "PyQt6", "PySide2", "PySide6", "sqlalchemy", "psycopg2", "psycopg2-binary", "pymysql",
    "fsspec", "s3fs", "gcsfs", "imageio-ffmpeg", "setuptools", "wheel", "pip",
    # Optional back ends a library imports only when a function needs one (skimage.restoration needs PyWavelets;
    # Keras 3 runs on whichever of these is installed; scikit-image's example data comes through pooch).
    "PyWavelets", "tensorflow", "tensorflow-cpu", "torch", "jax", "jaxlib", "pooch",
))

# Import roots that are not a package of their own (a part of one, or a namespace many packages share).
NOT_A_PACKAGE = frozenset({"mpl_toolkits", "pkg_resources", "_distutils_hack", "google", "azure", "ruamel"})

_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def pip_name(module: str) -> str | None:
    """The pip package a well-known import name comes from when the two differ (``skimage`` -> ``scikit-image``);
    ``None`` otherwise (the name is the package's own, or FI does not know it)."""
    return IMPORT_TO_PIP.get(module)


def import_names(dep: str, installed: Mapping[str, list[str] | None] | None = None) -> list[str] | None:
    """The names a requested package is imported by, when they are known: from the table above, or from the
    environment's own record of an installed package (``installed``, keyed by the PEP 503 name; see
    :func:`env_packages`). ``None`` when they are not known (a package FI does not know and that is not installed):
    the pip name is then not taken to be the import name."""
    name = requirement_name(dep)
    if not name:
        return None
    key = normalize(name)
    if key in _PIP_TO_IMPORTS:
        return list(_PIP_TO_IMPORTS[key])
    tops = (installed or {}).get(key)
    if tops:
        return [t for t in tops if _MODULE_RE.match(t) and not t.startswith("_")] or None
    return None


def warmup_modules(deps: Iterable[str], installed: Mapping[str, list[str] | None] | None = None) -> list[str]:
    """The modules the test import after installing imports: one per requested package whose import name is known
    (the table, or the installed package's own record). A package whose import name is not known is left out rather
    than imported under a guessed name (``scikit_image`` is not a module; the run itself will say if one is missing)."""
    out: list[str] = []
    for dep in deps:
        names = import_names(str(dep), installed)
        if not names:
            continue
        key = normalize(requirement_name(str(dep)))
        own = key.replace("-", "_")
        if len(names) == 1 or key in _PIP_TO_IMPORTS:  # the table lists the name every release has first
            pick = names[0]
        else:  # several top-level modules: the one named after the package, else none (which one is uncertain)
            pick = next((n for n in names if n.lower() == own), "")
        if pick and _MODULE_RE.match(pick) and pick not in out:
            out.append(pick)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# What the scripts import.
# ---------------------------------------------------------------------------------------------------------------------

def _optional_import_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            h.type is None or "ImportError" in ast.dump(h.type) or "ModuleNotFoundError" in ast.dump(h.type)
            for h in node.handlers
        ):
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    lines.add(child.lineno)
    return lines


def imported_modules(path: Path, *, optional: bool = True) -> set[str] | None:
    """The absolute imports in one script, as full dotted names (``from a.b import c`` gives ``a.b`` and ``a.b.c``).
    ``optional=False`` leaves out an import inside ``try: ... except ImportError``. ``None`` when the file cannot be
    read or parsed (so a caller can tell "imports nothing" from "could not tell")."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return None
    skip = set() if optional else _optional_import_lines(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) in skip:
            continue
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names if a.name != "*")
    return names


def third_party(names: Iterable[str], local: Iterable[str]) -> set[str]:
    """The top-level names of ``names`` that are neither the standard library nor the quest's own modules."""
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    own = set(local)
    tops = {n.split(".")[0] for n in names}
    return {n for n in tops if n not in stdlib and n not in own and n != "__future__"}


_SKIP_DIRS = {".git", "__pycache__", "run_output", "output", ".venv", "node_modules"}
_SOURCE_ROOTS = {"src", "lib"}  # folders a project puts on the path, whose modules are then imported by name
# More files than this and the scan stops; a caller that sees the cap reached cannot tell what is used.
SOURCE_LIMIT = 400


def code_sources(code_dir: Path, *, limit: int = SOURCE_LIMIT) -> list[Path]:
    """The quest's own scripts: every ``.py`` under ``code/`` (a multi-module project keeps modules in sub-folders)
    except FI's own ``run.py``. At most ``limit`` files."""
    return _py_files([(Path(code_dir), True)], limit)


def skill_sources(skills: Iterable[Any], *, limit: int = SOURCE_LIMIT) -> list[Path]:
    """The ``.py`` files of the selected skills: a library skill's are imported by the experiment, a tool skill's are
    run by it, so what either imports is something the run can need."""
    return _py_files([(Path(s.path), False) for s in skills if getattr(s, "path", None)], limit)


def _py_files(roots: list[tuple[Path, bool]], limit: int) -> list[Path]:
    out: list[Path] = []
    for root, is_code in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).parts
            if any(p in _SKIP_DIRS for p in rel[:-1]) or (is_code and rel == ("run.py",)):
                continue
            if path not in out:
                out.append(path)
            if len(out) >= limit:
                return out
    return out


def beside(script: Path) -> set[str]:
    """The modules next to a script, which it imports by name when it is run itself (``python tools/cli.py`` puts
    ``tools/`` first on the path, so its ``import helpers`` is ``tools/helpers.py``). None for a module inside a
    package (a folder with ``__init__.py``): it is imported, not run, and its ``import optuna`` is the library."""
    folder = Path(script).parent
    if (folder / "__init__.py").is_file():
        return set()
    try:
        return {p.stem for p in folder.glob("*.py")} | {
            p.name for p in folder.iterdir() if p.is_dir() and p.name not in _SKIP_DIRS}
    except OSError:
        return set()


def third_party_of(scripts: Iterable[Path], local: Iterable[str], *, optional: bool = True) -> set[str]:
    """The top-level third-party names the scripts import, each script's own neighbours counted as local to it."""
    own = set(local)
    out: set[str] = set()
    for script in scripts:
        out |= third_party(imported_modules(script, optional=optional) or set(), own | beside(script))
    return out


def local_module_names(code_dir: Path) -> set[str]:
    """The names the quest's own ``code/`` folder can be imported by: its scripts and folders at the top, and the
    modules and folders directly inside a source folder a project puts on the path (``src/`` or ``lib/`` without an
    ``__init__.py``: ``code/src/community.py`` is the quest's ``community``, not python-louvain). A module in any other
    folder is not: ``code/mypkg/optuna.py`` or ``code/utils/optuna.py`` does not hide the optuna library."""
    code_dir = Path(code_dir)
    if not code_dir.is_dir():
        return set()
    names = {p.stem for p in code_dir.glob("*.py")}
    for folder in code_dir.iterdir():
        if not folder.is_dir() or folder.name in _SKIP_DIRS:
            continue
        names.add(folder.name)
        if folder.name in _SOURCE_ROOTS and not (folder / "__init__.py").is_file():
            names |= {p.stem for p in folder.glob("*.py")}
            names |= {p.name for p in folder.iterdir() if p.is_dir() and p.name not in _SKIP_DIRS}
    return names


# ---------------------------------------------------------------------------------------------------------------------
# The quest environment's own record of its packages.
# ---------------------------------------------------------------------------------------------------------------------

_ENV_QUERY = r'''
import json, sys, importlib.util
if sys.path and sys.path[0] in ("", "."):
    del sys.path[0]  # "-c" puts FI's working folder first: a stray serial/ or yaml.py there is not the environment's
from importlib import metadata
req = json.loads(sys.argv[1])
dists = {}
for n in req["dists"]:
    try:
        d = metadata.distribution(n)
    except Exception:
        dists[n] = None
        continue
    tops = []
    try:
        txt = d.read_text("top_level.txt")
    except Exception:
        txt = None
    if txt:
        tops = [t.strip().replace("/", ".") for t in txt.split() if t.strip()]
    else:
        for f in (d.files or []):
            parts = f.parts
            if not parts or parts[0] in ("..", "bin", "Scripts") or parts[0].endswith((".dist-info", ".egg-info", ".data", ".pth")):
                continue
            if len(parts) > 1:
                if parts[0].isidentifier():  # "numpy.libs/", "foo-1.0.data/" are not packages
                    tops.append(parts[0])
            elif parts[0].endswith((".py", ".pyd", ".so")):
                tops.append(parts[0].split(".")[0])
    dists[n] = sorted(set(tops))
present = []
for m in req["modules"]:
    try:
        if importlib.util.find_spec(m) is not None:
            present.append(m)
    except Exception:
        pass
print(json.dumps({"dists": dists, "present": present}))
'''


@dataclass
class EnvInfo:
    """What the environment says: each asked package's top-level modules (``None``: not installed), keyed by the PEP
    503 name, and which of the asked modules it can import."""

    dists: dict[str, list[str] | None] = field(default_factory=dict)
    present: set[str] = field(default_factory=set)


def env_packages(python: Path | str, dists: Iterable[str], modules: Iterable[str] = ()) -> EnvInfo | None:
    """Ask the quest's Python which modules the named packages provide and which of ``modules`` it can import (one
    short subprocess, like :func:`core.code_project.pin`). ``None`` when it cannot be asked."""
    import subprocess

    names = sorted({requirement_name(d) for d in dists if requirement_name(d)})
    mods = sorted({m for m in modules if _MODULE_RE.match(m)})
    if not names and not mods:
        return EnvInfo()
    try:
        done = subprocess.run([str(python), "-c", _ENV_QUERY, json.dumps({"dists": names, "modules": mods})],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if done.returncode != 0:
            return None
        data = json.loads(done.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    raw = data.get("dists") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return None
    return EnvInfo(
        dists={normalize(k): (list(v) if isinstance(v, list) else None) for k, v in raw.items()},
        present={str(m) for m in data.get("present") or []},
    )


# ---------------------------------------------------------------------------------------------------------------------
# What to install: the requested packages the scripts use, and the well-known ones they import unrequested.
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class InstallPlan:
    install: list[str]                                             # what to ask pip for, in order
    unused: list[str] = field(default_factory=list)                # requested, no script uses it: not installed
    added: list[tuple[str, str]] = field(default_factory=list)     # (import name, pip name): imported, not requested
    renamed: list[tuple[str, str]] = field(default_factory=list)   # (requested, asked of pip): an import name given
    env: EnvInfo | None = None                                     # the environment's answer (None: not asked)
    all_because: str = ""                                          # why everything listed is installed, if it is


# Import names that are a different, unrelated package's name on PyPI too, so a request for one is taken as meant.
_OWN_PACKAGE_TOO = frozenset({"attr", "community", "jwt"})


def _pip_names_for_import_names(
    deps: list[str], *, keep: Iterable[str] = (),
) -> tuple[list[str], list[tuple[str, str]]]:
    """A requested name that is a well-known import name (``sklearn``, ``skimage``, ``cv2``, ``PIL``) is asked of pip
    under its package's name: under the import name pip finds nothing, or a stub that refuses to install. A name in
    ``keep`` (a selected skill's) is left as it is."""
    by_import = {normalize(m): p for m, p in IMPORT_TO_PIP.items() if m not in _OWN_PACKAGE_TOO and "." not in m}
    kept = {normalize(k) for k in keep}
    out: list[str] = []
    seen: set[str] = set()
    renamed: list[tuple[str, str]] = []
    for dep in deps:
        name = requirement_name(dep)
        pip = by_import.get(normalize(name)) if name and normalize(name) not in kept else None
        if pip:
            new = pip + dep[len(name):] if dep.startswith(name) else pip
            renamed.append((dep, new))
            dep, name = new, pip
        key = normalize(name or dep)
        if key not in seen:  # the first request for a package wins, as in split_deps
            seen.add(key)
            out.append(dep)
    return out, renamed


def _may_provide(dep: str, module: str, pip: str) -> bool:
    """Whether a request whose modules FI cannot tell might provide ``module`` (whose usual package is ``pip``): a
    URL or path with no name, a name sharing the package's first word (``opencv-contrib-python-rolling`` and
    ``opencv-python``, ``pillow-simd`` and ``Pillow``), or a name with the module's as one of its words."""
    name = requirement_name(dep)
    if not name:
        return True
    key, words = normalize(name), normalize(pip).split("-")
    n = 2 if words[0] in {"python", "scikit"} else 1  # a first word many packages share: compare two
    same_word = key.split("-")[:n] == words[:n]
    return same_word or module.split(".")[-1].lower() in key.split("-")  # a whole word: yamllint is not yaml


def _mentions(text: str, word: str) -> bool:
    return bool(word) and re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(word)}(?![A-Za-z0-9_-])", text, re.I) is not None


def plan_installs(
    deps: Iterable[str], sources: Iterable[Path], *, local: Iterable[str], python: Path | str | None,
    keep: Iterable[str] = (), skill_files: Iterable[Path] = (), keep_listed: bool = False,
) -> InstallPlan:
    """Which of the requested ``deps`` to install, checked against the scripts in ``sources``.

    A requested package is left out only when FI can tell it is unused and not there already: its import names are
    known from the table, the environment does not have it installed (or was not asked, as in a container), no script
    imports any of them (an optional ``try: import`` counts as a use), the scripts never mention the package or module
    by name (``engine="openpyxl"``, ``importlib.import_module("fitz")``, ``subprocess.run(["pytest"])`` all count),
    and it is not a package other libraries load by themselves (openpyxl, pyarrow, kaleido, scipy, PyWavelets, a Qt
    binding, ...). Anything uncertain is installed as before: no scripts, a script that does not parse, a package whose
    import name is unknown, a name in ``keep`` (a selected skill's), and every request when ``keep_listed`` (the last
    run failed on an import: a package left out once is never left out again on the same guess).

    A well-known package whose import name differs from its pip name (``import skimage``) that a script imports, that
    no request covers and that the environment cannot already import, is added under its pip name. ``python`` is the
    quest's interpreter to ask (``None``: not asked, as in a container; then nothing is added). ``skill_files`` are the
    selected skills' files, which the experiment imports or runs: what they import counts as used too (one that does
    not parse cannot be imported either, so it is skipped rather than making everything count)."""
    keep = list(keep)
    deps, renamed = _pip_names_for_import_names([str(d).strip() for d in deps if str(d).strip()], keep=keep)
    sources = list(sources)
    library = list(skill_files)
    parsed = [imported_modules(p) for p in sources]
    everything = InstallPlan(install=list(deps), renamed=renamed)  # cannot tell what the scripts use: all of it
    why = ("there is no script to check the list against" if not sources
           else "there are too many scripts to check" if len(sources) >= SOURCE_LIMIT or len(library) >= SOURCE_LIMIT
           else "a script has an error Python cannot parse, so what it imports is not known"
           if any(p is None for p in parsed) else "")
    if why:
        everything.all_because = why
        return everything
    lib_parsed = [imported_modules(p) for p in library]
    imported: set[str] = set().union(*parsed, *(p for p in lib_parsed if p is not None))
    tops = third_party_of(sources, local)
    try:
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in [*sources, *library])
    except OSError:
        everything.all_because = "a script could not be read, so what it imports is not known"
        return everything

    def is_imported(module: str) -> bool:
        return any(n == module or n.startswith(module + ".") for n in imported)

    # Well-known modules a script imports under a name its pip package does not have.
    candidates = sorted(m for m in IMPORT_TO_PIP if is_imported(m) and m.split(".")[0] in tops)
    env = env_packages(python, deps, candidates) if python is not None else None
    if python is not None and env is None:
        # The environment could not be asked: an installed package cannot be told from a missing one.
        everything.all_because = "the quest's Python could not be asked which packages it already has"
        return everything
    installed = env.dists if env else None

    kept_names = {normalize(requirement_name(k) or k) for k in keep}
    install: list[str] = []
    unused: list[str] = []
    covered: set[str] = set()
    unknown: list[str] = []  # kept requests whose modules FI cannot tell
    for dep in deps:
        name = requirement_name(dep)
        key = normalize(name) if name else ""
        names = import_names(dep, installed)
        used = (
            keep_listed
            or not name                     # a URL or a path: not FI's to judge
            or names is None                # import name unknown: installed, as before
            # Already installed: leaving it out saves nothing, and it would be missing from requirements.txt and the
            # lock file (a package the code needs only through another one, as torch under transformers).
            or bool(installed is not None and installed.get(key) is not None)
            or key in kept_names
            or key in _LOADED_BY_OTHERS
            or any(is_imported(n) for n in names)
            or _mentions(text, name) or _mentions(text, key)
            or any(_mentions(text, n) for n in names)
        )
        if used:
            install.append(dep)
            covered.update(names or [])
            if names is None:
                unknown.append(dep)
        else:
            unused.append(dep)
    added: list[tuple[str, str]] = []
    if env is not None:
        have = {normalize(requirement_name(d) or d) for d in install}
        for module in candidates:
            pip = IMPORT_TO_PIP[module]
            if module in covered or normalize(pip) in have or module in env.present:
                continue
            if any(_may_provide(d, module, pip) for d in unknown):
                continue  # a request FI cannot read may be this module already (a git build, another build of it)
            added.append((module, pip))
            install.append(pip)
            have.add(normalize(pip))
    return InstallPlan(install=install, unused=unused, added=added, renamed=renamed, env=env)
