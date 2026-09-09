"""Import a skill written for another agent.

Claude Code, and the several tools that copied its shape, store a skill as a
directory holding ``SKILL.md`` with YAML front matter::

    ---
    name: new-feature
    description: Feature development checklist
    ---
    Follow this checklist when adding new features.

FI's envelope is the same directory-plus-``SKILL.md`` shape, so that knowledge
transfers directly. What does **not** transfer is the promotion gate: those
skills carry no executable check, so an imported one lands ``UNTESTED`` and
cannot be used until someone writes one.

That is the gate working, not a gap to paper over. A checklist for adding a
feature is genuine knowledge and worth importing; it is not evidence that a
tool still behaves the way the checklist assumes. Importing preserves the
prose and states plainly what is still missing.

Front matter is preserved rather than stripped: ``description`` is what a
future retrieval step will match on, and discarding it on import would lose
the one field the source author wrote specifically to make the skill findable.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from core.skills.base import PROVENANCE_JSON, SELFTEST_PY, SKILL_MD

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.S)

#: Files worth carrying across. Anything else in a foreign skill directory
#: (agent-specific hooks, lockfiles) is left behind rather than imported into
#: a shape it was not written for.
_CARRY = ("SKILL.md", "api_surface.md", "reference.md", "README.md")
_CARRY_DIRS = ("scripts", "references", "assets", "examples")


@dataclass
class Imported:
    path: Path
    name: str
    source: Path
    description: str
    carried: list[str]
    has_selftest: bool

    @property
    def promotable(self) -> bool:
        return self.has_selftest


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML front matter from the body.

    Deliberately a small hand parser rather than a YAML dependency: the front
    matter of an agent skill is flat ``key: value`` lines, and a malformed
    block should degrade to "no metadata" rather than raise on import.
    """
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        v = value.strip().strip("'\"")
        meta[key.strip()] = v
    return meta, text[m.end():]


def _find_skill_md(source: Path) -> Path | None:
    if source.is_file():
        return source if source.name.lower().endswith(".md") else None
    candidate = source / SKILL_MD
    return candidate if candidate.is_file() else None


def import_skill(
    source: Path,
    dest_root: Path,
    *,
    name: str = "",
    overwrite: bool = False,
    domains: list[str] | None = None,
) -> Imported:
    """Bring a foreign skill into FI's envelope.

    ``source`` is a ``SKILL.md`` or the directory holding one. The name comes
    from ``--name``, else the front matter, else the directory name — in that
    order, because the caller is the only one who knows what it should be
    called *here*.
    """
    source = Path(source).expanduser()
    skill_md = _find_skill_md(source)
    if skill_md is None:
        raise FileNotFoundError(
            f"No SKILL.md at {source}. Point at a skill directory or the "
            "SKILL.md itself."
        )

    raw = skill_md.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    src_dir = skill_md.parent
    resolved = (
        name.strip()
        or meta.get("name", "").strip()
        or src_dir.name
    )
    if not resolved:
        raise ValueError("Could not determine a name; pass one explicitly.")

    dest = dest_root / resolved
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"{dest} already exists — pass overwrite to replace it, and note "
            "that replacing a skill lapses its approval"
        )
    dest.mkdir(parents=True, exist_ok=True)

    # Keep the front matter: `description` is what makes a skill findable,
    # and it was written by someone who knew the skill.
    header = ["---", f"name: {resolved}"]
    if meta.get("description"):
        header.append(f"description: {meta['description']}")
    header += ["---", ""]
    (dest / SKILL_MD).write_text(
        "\n".join(header) + body.lstrip("\n"), encoding="utf-8"
    )

    carried = [SKILL_MD]
    if src_dir.is_dir():
        for fname in _CARRY:
            if fname == SKILL_MD:
                continue
            f = src_dir / fname
            if f.is_file():
                shutil.copy2(f, dest / fname)
                carried.append(fname)
        for dname in _CARRY_DIRS:
            d = src_dir / dname
            if d.is_dir():
                shutil.copytree(d, dest / dname, dirs_exist_ok=True)
                carried.append(f"{dname}/")

    (dest / PROVENANCE_JSON).write_text(
        json.dumps(
            {
                "origin": "imported",
                # Domain tags route the catalogue and grant nothing, so
                # unhashed provenance is the right home for them. An untagged
                # skill counts as general and is always a candidate — an
                # import that forgets its tags therefore over-offers rather
                # than silently withdrawing a capability.
                "domains": [d.strip().lower() for d in (domains or []) if d.strip()],
                "imported_from": str(skill_md),
                "imported_at": time.strftime("%Y-%m-%d"),
                "source_frontmatter": meta,
                "taught_by_projects": [],
                "result_assertions": [],
                "_todo": (
                    "Imported skills carry no executable check. Add selftest.py "
                    "proving the tool this describes is present and behaves as "
                    "the instructions assume — that is what stops a quest "
                    "wasting a run on a tool that moved, changed or vanished."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return Imported(
        path=dest,
        name=resolved,
        source=skill_md,
        description=meta.get("description", ""),
        carried=carried,
        has_selftest=(dest / SELFTEST_PY).is_file(),
    )


def discover_foreign(roots: list[Path] | None = None) -> list[tuple[str, Path, str]]:
    """Find skills belonging to other agents, for `--import-skill` to name.

    Read-only: these are *not* added to FI's discovery. A checklist telling an
    agent to run a slash command has no business being injected into an
    experiment-design prompt, and it could never be promoted anyway. Listing
    them is so a person can choose one to adapt.
    """
    if roots is None:
        home = Path.home()
        roots = [home / ".claude" / "skills"]
        # Project-level skills for every checkout beside this one.
        for parent in (Path.cwd().parent, home / "dev"):
            if parent.is_dir():
                roots += sorted(parent.glob("*/.claude/skills"))

    found: list[tuple[str, Path, str]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            md = d / SKILL_MD
            if not md.is_file() or md in seen:
                continue
            seen.add(md)
            try:
                meta, _ = parse_frontmatter(md.read_text(encoding="utf-8"))
            except OSError:
                meta = {}
            found.append((d.name, d, meta.get("description", "")))
    return found
