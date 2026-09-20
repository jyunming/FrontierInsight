"""Approved external skills, made reachable inside the Docker sandbox.

A skill another agent installed is read where it lives (``~/.claude/skills/x``).
With ``execution.sandbox: docker`` the experiment runs in a container that sees
the quest and nothing else, so a skill's scripts and data were named in the
prompt by a host path that does not exist there. This module decides which skill
folders are bind-mounted, read-only, at a fixed container path, and says where.

What is mounted is the smallest thing that answers the need:

* only a skill that is **external**, **loadable** (a person approved its exact
  content, which is what ``SkillState.loadable`` means) and **selected** for the
  quest by the caller. Never a skills folder, never a home directory, never a
  tool's configuration folder;
* only the skill's own folder, at ``/fi-skills/<safe-name>``, ``ro``.

A folder is refused, with the reason, when any of these holds:

* it is a symbolic link (``Path.is_symlink``);
* it resolves anywhere but a direct child of the folder it was found in (this
  also catches a Windows junction, which ``is_symlink`` does not report);
* it is not a directory holding a regular ``SKILL.md``;
* it is the home directory or one of its ancestors, or a dot-folder directly in
  the home directory (``~/.claude``, ``~/.ssh``: tool configuration and
  credentials);
* its path cannot be written into a ``host:container:mode`` bind string without
  ambiguity (a colon that is not the drive letter's, a control character, a
  network path).

Links *inside* a mounted folder are not followed by the mount: a link is
resolved by the container against the container's own filesystem, so one that
points out of the skill dangles there instead of reaching the host.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Iterable

from core.skills.base import SKILL_MD, SkillState

#: Where the sandbox sees skills. One folder per skill below it.
CONTAINER_ROOT = "/fi-skills"

_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_EXTENDED_PREFIX = "\\\\?\\"


def container_dir_name(name: str) -> str:
    """The folder name a skill gets under ``CONTAINER_ROOT``.

    A name that is already plain (letters, digits, ``.``, ``_``, ``-``, not
    starting or ending with a separator, no ``..``) is kept. Any other name is
    reduced to plain characters and given a short digest of the original, so two
    different names never share a folder and the result is stable from run to
    run.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+|\.{2,}", "_", name).strip("._-")[:64]
    if safe and safe == name:
        return safe
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'skill'}-{digest}"


def bind_host_path(path: "str | PurePath") -> str:
    """The host side of a ``host:container:mode`` bind string.

    The spelling is the one the quest's own ``/work`` mount already uses
    (``str`` of the resolved path, so a drive-letter path stays ``C:\\...`` on
    Windows). What this adds is the refusal of what would corrupt or redirect
    the string: the bind string is split on ``:``, so the only colon allowed is
    a drive letter's. Raises ``ValueError`` with the reason.
    """
    s = str(path)
    if s.startswith(_EXTENDED_PREFIX):
        rest = s[len(_EXTENDED_PREFIX):]
        if not _DRIVE.match(rest):
            raise ValueError("a network (UNC) path cannot be mounted")
        s = rest
    elif s.startswith("\\\\"):
        raise ValueError("a network (UNC) path cannot be mounted")
    tail = s[2:] if _DRIVE.match(s) else s
    if ":" in tail:
        raise ValueError("the path contains a ':', which a bind mount cannot express")
    if any(ord(c) < 32 for c in s):
        raise ValueError("the path contains a control character")
    return s


def _real(p: Path) -> Path:
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        return p


def _fold(name: str) -> str:
    return os.path.normcase(name)


def check_folder(folder: Path, name: str, *, home: Path | None = None) -> tuple[Path | None, str]:
    """Decide whether ``folder`` (a skill folder as discovered, ``<root>/<name>``)
    may be mounted. Returns ``(resolved folder, "")`` or ``(None, reason)``.

    The declared root is ``folder.parent``: the folder the skill was found in.
    """
    if folder.name != name:
        return None, "its folder name is not the skill's name"
    try:
        if folder.is_symlink():
            return None, "the skill folder is a symbolic link"
    except OSError as exc:
        return None, f"the skill folder cannot be examined ({exc})"
    try:
        real = folder.resolve(strict=True)
        root = folder.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return None, f"the skill folder cannot be resolved ({exc})"
    # A folder that is not itself a link resolves to <resolved root>/<name>. Any
    # other answer means the path leads somewhere else (a Windows junction is
    # the case is_symlink does not report).
    if real.parent != root or _fold(real.name) != _fold(folder.name):
        return None, "the skill folder resolves outside its skills folder"
    if not real.is_dir():
        return None, "the skill folder is not a directory"
    if not (real / SKILL_MD).is_file():
        return None, "the skill folder has no regular SKILL.md"
    try:
        home_real = _real(home if home is not None else Path.home())
    except RuntimeError:
        return None, "the home directory cannot be determined, so the folder cannot be checked against it"
    if real == home_real or real in home_real.parents:
        return None, "the skill folder is the home directory or one of its parents"
    if real.parent == home_real and real.name.startswith("."):
        return None, "the skill folder is a tool's configuration folder in the home directory"
    return real, ""


@dataclass(frozen=True)
class SkillMount:
    """One skill folder and where the container sees it."""

    name: str
    host: Path            # the resolved real folder, the source of the bind
    container: str        # POSIX path in the container, under CONTAINER_ROOT
    declared: Path        # the path the skill was discovered at

    @property
    def bind_host(self) -> str:
        return bind_host_path(self.host)

    @property
    def volume(self) -> dict[str, str]:
        """The docker-py ``volumes`` value: read-only, always."""
        return {"bind": self.container, "mode": "ro"}

    @property
    def bind_arg(self) -> str:
        """The same mount as ``docker run -v`` spells it."""
        return f"{self.bind_host}:{self.container}:ro"

    def still_safe(self, *, home: Path | None = None) -> bool:
        """Checked again when a container is created: the folder must still be
        the one that was approved and planned, not a link put there since."""
        real, _ = check_folder(self.declared, self.name, home=home)
        return real is not None and real == self.host

    def translate(self, text: str) -> str:
        """``text`` with this skill's host folder replaced by its container
        path. A path that only starts like the folder (``foo-bar`` for ``foo``,
        ``foo.py``) is left alone, and the segments after the folder keep
        their place with ``/`` separators."""
        spellings = {
            str(self.declared), str(self.host),
            self.declared.as_posix(), self.host.as_posix(),
        }
        for spelling in sorted(spellings, key=len, reverse=True):
            pattern = (
                re.escape(spelling)
                + r"(?![\w-])(?!\.\w)"
                + r"(?P<rest>(?:[\\/][^\s`'\"<>|*?,;()\[\]{}]*)*)"
            )
            text = re.sub(
                pattern,
                lambda m: self.container + m.group("rest").replace("\\", "/"),
                text,
            )
        return text


@dataclass
class MountPlan:
    """Which selected skills are mounted, and which are not and why."""

    mounts: dict[str, SkillMount] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)

    def container_path(self, name: str) -> str | None:
        m = self.mounts.get(name)
        return m.container if m else None


def plan_mounts(states: Iterable[SkillState], *, home: Path | None = None) -> MountPlan:
    """The mounts for the skills a quest selected. ``states`` are what
    ``loadable_skills`` returned; only an external one is considered (FI's own
    skills are not this feature's concern), and only a loadable one is
    mounted, so a skill that was never approved, or whose content changed since
    it was, is refused here even if a caller passes it in."""
    plan = MountPlan()
    taken: dict[str, str] = {}
    for st in states:
        skill = st.skill
        if not skill.external:
            continue
        if not st.loadable:
            plan.refused[skill.name] = f"it is not approved ({st.status.value})"
            continue
        real, why = check_folder(skill.path, skill.name, home=home)
        if real is None:
            plan.refused[skill.name] = why
            continue
        try:
            bind_host_path(real)
        except ValueError as exc:
            plan.refused[skill.name] = str(exc)
            continue
        leaf = container_dir_name(skill.name)
        if leaf in taken:  # two names reduced to one folder name (astronomically unlikely)
            leaf = f"{leaf}-{hashlib.sha256(str(real).encode('utf-8')).hexdigest()[:8]}"
        taken[leaf] = skill.name
        plan.mounts[skill.name] = SkillMount(
            name=skill.name, host=real, container=f"{CONTAINER_ROOT}/{leaf}",
            declared=skill.path,
        )
    return plan
