"""Remove a skill from FI completely.

``--revoke-skill`` only withdraws approval: the skill stays on disk and is listed
again as proposed. Removing goes further, and what it can do depends on who owns
the files:

* a skill in FI's own folder is deleted, with its approval and recorded passes;
* a skill another agent installed is read in place and FI never writes into that
  folder, so it is hidden from FI instead (the files are left untouched) and can
  be brought back with ``restore_external``;
* a skill a pip package ships cannot be deleted from here.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.skills import approval, selftest_cache
from core.skills.base import Skill

_HIDDEN_FILE = "skills_removed_external.json"


def hidden_path() -> Path:
    """Where the names of hidden external skills are kept, beside the approval ledger."""
    return approval.ledger_path().parent / _HIDDEN_FILE


def hidden_external() -> set[str]:
    try:
        data = json.loads(hidden_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    names = data.get("names") if isinstance(data, dict) else None
    return {n for n in names if isinstance(n, str)} if isinstance(names, list) else set()


def _save_hidden(names: set[str]) -> None:
    p = hidden_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"names": sorted(names)}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


@dataclass(frozen=True)
class RemovalResult:
    ok: bool
    message: str
    deleted: Path | None = None
    hidden: bool = False


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attrs = os.lstat(path).st_file_attributes  # Windows only; Python 3.11 has no Path.is_junction
    except (OSError, AttributeError):
        return False
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _make_writable_and_retry(func: Any, path: str, _exc: Any) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _delete_tree(path: Path) -> None:
    if _is_link(path):
        # Only the link goes; what it points to is not FI's to delete.
        if path.is_symlink():
            path.unlink()
        else:
            os.rmdir(path)
        return
    if "onexc" in shutil.rmtree.__code__.co_varnames:
        shutil.rmtree(path, onexc=_make_writable_and_retry)
    else:
        shutil.rmtree(path, onerror=_make_writable_and_retry)


def _entry_point_dist(name: str) -> str:
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="fi.skills"):
            if ep.name == name and getattr(ep, "dist", None) is not None:
                return str(ep.dist.name)
    except Exception:  # noqa: BLE001 - best effort, the message has a generic form
        pass
    return ""


def _forget(skill: Skill) -> bool:
    withdrawn = approval.revoke(skill.ledger_name)
    selftest_cache.forget_skill(skill.name)
    return withdrawn


def remove_skill(skill: Skill) -> RemovalResult:
    from core.skills.registry import skills_root

    name = skill.name
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        return RemovalResult(False, f"Refusing to remove {name!r}: not a plain skill name.")

    if skill.external:
        withdrawn = _forget(skill)
        names = hidden_external()
        names.add(name)
        _save_hidden(names)
        extra = ", approval withdrawn" if withdrawn else ""
        return RemovalResult(
            True,
            f"Removed skill {name} from FI: hidden{extra}. Its files at {skill.path} belong to another "
            f"tool and were left as they are. To bring it back: python launch.py --restore-skill {name}",
            hidden=True,
        )

    if skill.source != "filesystem":
        dist = _entry_point_dist(name)
        how = f"pip uninstall {dist}" if dist else "pip uninstall the package that ships it"
        return RemovalResult(
            False,
            f"Skill {name} is installed by a Python package, so FI cannot delete it. Run: {how}",
        )

    root = skills_root()
    target = root / name
    try:
        inside = target.parent.resolve() == root.resolve() and skill.path.parent.resolve() == root.resolve()
    except OSError:
        inside = False
    if not inside or skill.path.name != name or not (target.exists() or _is_link(target)):
        return RemovalResult(False, f"Refusing to delete {skill.path}: it is not a skill folder of {root}.")

    # Approval goes first: a folder that survives a failed delete is then merely proposed again, never
    # left approved with half its files gone.
    withdrawn = _forget(skill)
    try:
        _delete_tree(target)
    except OSError as e:
        return RemovalResult(False, f"Could not delete {target}: {e}. Close whatever is using it and try again.")
    extra = ", approval withdrawn" if withdrawn else ""
    return RemovalResult(True, f"Removed skill {name}: deleted {target}{extra}.", deleted=target)


def restore_external(name: str) -> bool:
    """Bring a hidden external skill back. False when it was not hidden."""
    names = hidden_external()
    if name not in names:
        return False
    names.discard(name)
    _save_hidden(names)
    return True
