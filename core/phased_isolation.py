"""Whether the rows held back for a confirm run were out of exploration's reach (``core/phased.py``).

Keeping the held-back part outside the quest folder (``<output_dir>/_held_back/<quest_id>/``) is not by itself
isolation: a script that runs with the quest folder as its working directory can open ``../_held_back/...``. A
held-back confirm counts as on unseen data only when one of two things held while exploration ran:

- :data:`DOCKER`: every run of the quest's code was in a container that mounts the quest folder (which holds only
  exploration's part during exploration, and only the held-back part during the confirm run) and nothing that holds the
  kept files (:func:`docker_mounts_clear`).
- :data:`ENCRYPTED`: without a container, the held-back part, and the whole file it was taken from, were written
  encrypted (:func:`seal`, Fernet: AES with an HMAC, from the ``cryptography`` package) with a key made for that run of
  FI and kept only in FI's own memory, never on disk and never in the environment of the quest's code; and, before the
  confirm run, the quest's code was read for any path that leads out of the quest folder (:func:`scan`) and none was
  found. A scan is a reading of the code as it is before the confirm run for plain patterns (an absolute path, ``..`` or
  ``os.pardir``, a parent of the working folder or of the script past the quest folder, the home folder, the name
  ``_held_back``, a read of another program's memory), not proof: a path built from pieces at run time is not seen, and
  anything it finds keeps the result below publication-ready.

Otherwise the status is :data:`UNVERIFIED`, a ``publication_ready`` gap said in one plain sentence. A confirm run on new
seeds hides no data, and is not judged here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

DOCKER = "docker"
ENCRYPTED = "encrypted+scanned"
UNVERIFIED = "isolation_unverified"

#: The first bytes of a kept file written encrypted.
MAGIC = b"FI-HELD-BACK-ENCRYPTED/1\n"


def new_key() -> bytes:
    """A key for one run of FI. Kept by the engine in memory only."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key()


def seal(data: bytes, key: bytes | None) -> bytes:
    """``data`` encrypted with ``key`` (unchanged without one: a container keeps the files out of reach)."""
    if key is None:
        return data
    from cryptography.fernet import Fernet

    return MAGIC + Fernet(key).encrypt(data)


def sealed(data: bytes) -> bool:
    return data.startswith(MAGIC)


def unseal(data: bytes, key: bytes | None) -> bytes | None:
    """The plain bytes of a kept file, or None when it is encrypted and ``key`` is not the one it was written with (the
    run of FI that wrote it has ended: a killed run, or a background job that outlived it)."""
    if not sealed(data):
        return data
    if key is None:
        return None
    from cryptography.fernet import Fernet, InvalidToken

    try:
        return Fernet(key).decrypt(data[len(MAGIC):])
    except (InvalidToken, ValueError):
        return None


# ---- the scan of the quest's code ------------------------------------------------------------------------------------

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".ipynb_checkpoints"}
_SCRIPTS = {".py", ".sh", ".bash", ".bat", ".cmd", ".ps1", ".r", ".jl", ".m", ".pl", ".rb", ".js", ".ts"}
_ABSOLUTE = re.compile(r"^(?:/(?!dev/(?:null|stdout|stderr)\b)[A-Za-z0-9_.~-]|[A-Za-z]:[\\/]|\\\\|~(?:[\\/]|$))")
_UP = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")
_HELD = re.compile(r"_held_back", re.I)
_HOME_CALLS = {"expanduser", "home", "gethomedir"}
_HOME_VARS = {"HOME", "USERPROFILE", "HOMEPATH", "HOMEDRIVE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"}
_CWD_CALLS = {"cwd", "getcwd", "getcwdb"}


def _depth(path: Path, quest_root: Path) -> int:
    try:
        return len(path.relative_to(quest_root).parts) - 1
    except ValueError:
        return 0


def _parents(node: ast.AST) -> tuple[int, ast.AST]:
    """How many ``.parent`` / ``.parents[i]`` / ``os.path.dirname(...)`` steps lead up from the innermost expression."""
    count = 0
    while True:
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            count, node = count + 1, node.value
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents"
              and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)):
            count, node = count + node.slice.value + 1, node.value.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dirname"
              and node.args):
            count, node = count + 1, node.args[0]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                "resolve", "absolute"):
            node = node.func.value
        elif isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)) and node.args and (
                (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) in ("Path", "abspath", "realpath")):
            node = node.args[0]
        else:
            return count, node


def _is_file_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "__file__"


def _is_cwd(node: ast.AST) -> bool:
    """The working folder: ``Path.cwd()``, ``os.getcwd()``, ``Path()``, ``"."``."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _CWD_CALLS:
        return True
    if isinstance(node, ast.Call) and not node.args and (
            (node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")) in ("Path", "PurePath")):
        return True
    return isinstance(node, ast.Constant) and node.value in (".", "./", ".\\")


#: Names that read another process's memory (FI's own, where the key is): never needed by a study.
_MEMORY = {"ReadProcessMemory", "OpenProcess", "process_vm_readv", "NtReadVirtualMemory"}


def _docstrings(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _scan_python(path: Path, rel: str, depth: int) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return _scan_text(path, rel)
    docs = _docstrings(tree)
    hits: list[str] = []

    def hit(node: ast.AST, what: str) -> None:
        hits.append(f"{rel} line {getattr(node, 'lineno', '?')}: {what}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            text = node.value.strip()
            if _HELD.search(text):
                hit(node, "names the folder the held-back rows are kept in (_held_back)")
            elif re.match(r"^/proc/(?:\d|\{)", text) or text in _MEMORY:
                hit(node, "reads the memory of another program")
            elif "://" not in text and "\n" not in text and _ABSOLUTE.match(text):
                hit(node, f"opens an absolute path ({text[:60]!r})")
            elif "\n" not in text and _UP.search(text):
                hit(node, f"leads out of the folder with '..' ({text[:60]!r})")
        elif isinstance(node, ast.Name) and _HELD.search(node.id):
            hit(node, "names the folder the held-back rows are kept in (_held_back)")
        elif (isinstance(node, ast.Attribute) and node.attr in _MEMORY) or (
                isinstance(node, ast.Name) and node.id in _MEMORY):
            hit(node, "reads the memory of another program")
        elif isinstance(node, ast.Attribute) and node.attr == "pardir":
            hit(node, "leads out of the folder with os.pardir ('..')")
        elif isinstance(node, ast.Attribute) and node.attr in ("parent", "parents") or (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dirname"):
            count, base = _parents(node)
            if _is_file_name(base) and count > depth + 1:
                hit(node, "goes up from the script's own folder past the quest folder")
            elif _is_cwd(base) and count >= 1:
                hit(node, "goes up from the working folder (the quest folder) to the folder above it")
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in _HOME_CALLS:
                hit(node, "reads the home folder")
            elif name in ("getenv", "get") and node.args and isinstance(node.args[0], ast.Constant) \
                    and str(node.args[0].value).upper() in _HOME_VARS:
                hit(node, f"reads a folder outside the quest from the environment ({node.args[0].value})")
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and str(node.slice.value).upper() in _HOME_VARS and "environ" in ast.unparse(node.value):
            hit(node, f"reads a folder outside the quest from the environment ({node.slice.value})")
    return list(dict.fromkeys(hits))


def _scan_text(path: Path, rel: str) -> list[str]:
    hits: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [f"{rel}: could not be read, so it was not checked"]
    for number, line in enumerate(lines, start=1):
        code = line.split("#", 1)[0]
        if _HELD.search(code):
            hits.append(f"{rel} line {number}: names the folder the held-back rows are kept in (_held_back)")
        elif re.search(r"(?:^|[\s'\"=(])\.\.(?:[\\/]|$)", code):
            hits.append(f"{rel} line {number}: leads out of the folder with '..'")
        elif re.search(r"(?:^|[\s'\"=(])(?:/(?:home|Users|root|mnt|media|srv|opt|var|etc|tmp)\b|[A-Za-z]:[\\/]|~[\\/])",
                       code):
            hits.append(f"{rel} line {number}: opens an absolute path")
    return hits


def scan(quest_root: Path, *, folders: tuple[str, ...] = ("code",)) -> list[str]:
    """Plain lines, one for each place in the quest's code (``code/``, its package and scripts) that reads, or leads
    to, a path outside the quest folder. Empty: nothing found."""
    quest_root = Path(quest_root)
    hits: list[str] = []
    for folder in folders:
        base = quest_root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or any(part in _SKIP_DIRS for part in path.relative_to(base).parts):
                continue
            suffix = path.suffix.lower()
            if suffix not in _SCRIPTS:
                continue
            rel = path.relative_to(quest_root).as_posix()
            depth = _depth(path, quest_root)
            hits += _scan_python(path, rel, depth) if suffix == ".py" else _scan_text(path, rel)
    return hits


# ---- a container --------------------------------------------------------------------------------------------------


def docker_mounts_clear(executor: Any, quest_root: Path, store: Path) -> str:
    """Empty when every folder the container gets (``DockerExecutor._volumes``) leaves the kept files out; else why."""
    volumes = getattr(executor, "_volumes", None)
    if volumes is None:
        return "the run was not in a container"
    try:
        mounted = list(volumes(Path(quest_root).resolve()))
    except Exception as e:  # noqa: BLE001 -- a mount list that cannot be read is not shown to be clear
        return f"the folders given to the container could not be listed ({e!r})"
    store = Path(store).resolve()
    for host in mounted:
        h = Path(host).resolve()
        if h == store or h in store.parents or store in h.parents:
            return f"the container is given {h}, which holds the kept files"
    return ""


def describe(status: str) -> str:
    """The isolation status in a few plain words."""
    return {DOCKER: "exploration's code ran in a container that could not reach the held-back rows",
            ENCRYPTED: ("the held-back rows were kept encrypted with a key only FI held while exploration ran, and the "
                        "quest's code was read for paths out of the quest folder and none was found")}.get(
        status, "the held-back rows were not shown to be out of the reach of exploration's code")
