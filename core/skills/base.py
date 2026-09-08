"""What a skill is on disk, and what state it can be in.

A **skill** is what FI knows about driving one piece of scientific
software: when to use it, how to call it, an executable check that proves
it still works, and the range assertions its outputs must satisfy.

The envelope is the same at every maturity, so nothing has to migrate as
a skill hardens:

    skills/<name>/
      SKILL.md         always     — when to use it, workflow, gotchas
      api_surface.md   optional   — signatures fed to design / implement
      skill.py         optional   — executable entry points
      selftest.py      required for promotion
      provenance.json  always     — origin, and which projects taught it
      scripts/         optional   — executable code, Agent Skills layout
      references/      optional   — documentation, Agent Skills layout
      assets/          optional   — templates and data

The last three come from the `Agent Skills open standard
<https://agentskills.io/>`_, which the same directory-plus-``SKILL.md``
shape underlies. FI reads them so a skill written for another agent works
here unchanged; ``api_surface.md`` and ``skill.py`` remain FI's own names
for the single-file cases. Their *contents* are never injected into a
prompt — the standard loads them on demand at execution, and so does FI,
because a references directory can be larger than the quest itself.

Maturity is derived from which files exist, not declared:

    notes    SKILL.md only            — FI generates code, guided by prose
    guided   + api_surface.md         — generation guided by real signatures
    adapter  + skill.py               — FI calls the code instead

Status is a different axis and is never derived from the files, because
it encodes a human decision:

    QUARANTINED  self-test exists and fails — never loaded
    UNTESTED     no self-test — cannot be promoted
    PROPOSED     self-test passes, awaiting approval
    TRUSTED      self-test passes and a person approved *this content*

Approval binds to a content hash rather than a name. A skill that gets
re-distilled from new work is a different skill wearing the same name,
and inheriting the old approval would turn the human gate into a
one-time rubber stamp. See ``approval.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

SKILL_MD = "SKILL.md"
API_SURFACE_MD = "api_surface.md"
SKILL_PY = "skill.py"
SELFTEST_PY = "selftest.py"
PROVENANCE_JSON = "provenance.json"

#: Agent Skills standard directories.
SCRIPTS_DIR = "scripts"
REFERENCES_DIR = "references"
ASSETS_DIR = "assets"

#: The content hash covers *everything* in the skill directory except what
#: is listed here. It is an exclusion list rather than an allow list on
#: purpose: the allow list version hashed four fixed filenames while the
#: importer copied whole ``scripts/`` and ``references/`` trees in beside
#: them, so editing an approved skill's script left its approval intact.
#: Anything executable that arrives must be covered by default, or the
#: human gate is only guarding the files someone remembered to name.
#:
#: ``provenance.json`` stays out because recording that a skill was used in
#: one more quest must not revoke its approval. Bytecode stays out because
#: running the self-test *creates* it inside the skill directory — hashing
#: it would lapse a skill's approval every time it was checked.
_HASH_EXCLUDED_NAMES = frozenset({PROVENANCE_JSON})
_HASH_EXCLUDED_DIRS = frozenset({"__pycache__", ".git", ".pytest_cache"})
_HASH_EXCLUDED_SUFFIXES = (".pyc", ".pyo")


class Kind(str, Enum):
    """What the skill is about operating.

    A skill is not only a verified physics kernel — it is knowledge of how to
    operate something. That something can be an importable library or an
    external tool, and the two need different instructions, different
    self-tests, and different language in the prompt.
    """

    LIBRARY = "library"   # importable; FI calls its functions
    TOOL = "tool"         # external binary or service; FI drives it


class Maturity(str, Enum):
    NOTES = "notes"
    GUIDED = "guided"
    ADAPTER = "adapter"


class Status(str, Enum):
    QUARANTINED = "quarantined"
    UNTESTED = "untested"
    PROPOSED = "proposed"
    TRUSTED = "trusted"

    @property
    def loadable(self) -> bool:
        """Only a trusted skill may enter a quest prompt.

        Deliberately a property on the enum rather than a check scattered
        through the engine: there is exactly one place to change if the
        policy ever moves, and no call site can forget it.
        """
        return self is Status.TRUSTED


@dataclass(frozen=True)
class Skill:
    """One skill as found on disk."""

    name: str
    path: Path
    source: str = "filesystem"  # or an entry-point group name

    # ---- envelope ----

    def file(self, filename: str) -> Path:
        return self.path / filename

    def has(self, filename: str) -> bool:
        return self.file(filename).is_file()

    @property
    def valid(self) -> bool:
        """A directory is a skill only if it carries instructions."""
        return self.has(SKILL_MD)

    @property
    def maturity(self) -> Maturity:
        if self.has(SKILL_PY):
            return Maturity.ADAPTER
        if self.has(API_SURFACE_MD):
            return Maturity.GUIDED
        return Maturity.NOTES

    @property
    def has_selftest(self) -> bool:
        return self.has(SELFTEST_PY)

    def _dir_listing(self, dirname: str) -> list[str]:
        d = self.path / dirname
        if not d.is_dir():
            return []
        try:
            out = []
            for p in d.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.path)
                if set(rel.parts[:-1]) & _HASH_EXCLUDED_DIRS:
                    continue
                if p.suffix.lower() in _HASH_EXCLUDED_SUFFIXES:
                    continue
                out.append(rel.as_posix())
        except OSError:
            return []
        return sorted(out)

    def bundled_scripts(self) -> list[str]:
        """Executable files shipped with the skill (standard ``scripts/``).

        Paths only. A quest is told these exist and where they are, never
        their contents: a bundled tool can be thousands of lines, and the
        point of selection is that a skill costs a bounded number of tokens.
        """
        return self._dir_listing(SCRIPTS_DIR)

    def reference_files(self) -> list[str]:
        """Documentation shipped with the skill (standard ``references/``).

        Paths only, for the same reason as ``bundled_scripts``. The standard
        calls this progressive disclosure and loads them at execution time;
        FI does the same by naming them and letting the generated code read
        what it needs.
        """
        return self._dir_listing(REFERENCES_DIR)

    @property
    def kind(self) -> "Kind":
        """Is this an importable library, or software FI drives from outside?

        The distinction changes what a quest should be told. "Call it instead
        of re-deriving the physics" is right for a library and meaningless for
        a command-line tool, which has no physics and cannot be imported.

        An explicit ``kind`` in provenance wins. Otherwise it is inferred from
        evidence rather than defaulted blindly: a skill carrying an API
        surface or executable entry points is describing something importable;
        anything else is describing something driven from outside.

        A standard-layout skill shipping only ``scripts/`` therefore infers
        TOOL, which is the right answer: a bundled script is something you
        run, not something you import, and telling a quest to import it would
        teach it to write code that cannot work.
        """
        declared = str(self.provenance().get("kind", "")).strip().lower()
        if declared in (k.value for k in Kind):
            return Kind(declared)
        if self.has(API_SURFACE_MD) or self.has(SKILL_PY):
            return Kind.LIBRARY
        return Kind.TOOL

    # ---- content ----

    def instructions(self) -> str:
        """The prose fed to design / implement. Empty when unreadable."""
        try:
            return self.file(SKILL_MD).read_text(encoding="utf-8")
        except OSError:
            return ""

    def api_surface(self) -> str:
        try:
            return self.file(API_SURFACE_MD).read_text(encoding="utf-8")
        except OSError:
            return ""

    def provenance(self) -> dict[str, Any]:
        try:
            data = json.loads(self.file(PROVENANCE_JSON).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def assertions(self) -> list[dict[str, Any]]:
        """Range assertions this skill's outputs must satisfy.

        Same shape as a design's ``result_assertions`` (see
        ``core.plausibility``). A skill knows its own valid domain, so it
        can supply these without the design having to restate them.
        """
        raw = self.provenance().get("result_assertions")
        return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []

    # ---- identity ----

    def hashable_files(self) -> list[tuple[str, Path]]:
        """Every file that defines behaviour, as (relative posix path, path).

        Sorted by the relative path rather than the absolute one so the
        digest does not depend on where the skill lives or on the platform's
        path separator.
        """
        out: list[tuple[str, Path]] = []
        try:
            for p in self.path.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.path)
                if set(rel.parts[:-1]) & _HASH_EXCLUDED_DIRS:
                    continue
                if rel.name in _HASH_EXCLUDED_NAMES:
                    continue
                if p.suffix.lower() in _HASH_EXCLUDED_SUFFIXES:
                    continue
                out.append((rel.as_posix(), p))
        except OSError:
            return []
        out.sort(key=lambda item: item[0])
        return out

    def content_hash(self) -> str:
        """Stable digest of the files that define behaviour.

        Any change to instructions, surface, code, test or bundled script
        produces a new hash and so lapses approval. Provenance updates and
        bytecode do not.

        The path is hashed alongside the bytes, so moving a script to a new
        name is a change even when no byte of its content differs.
        """
        h = hashlib.sha256()
        for rel, p in self.hashable_files():
            h.update(rel.encode("utf-8"))
            try:
                h.update(p.read_bytes())
            except OSError:
                h.update(b"\0")
        return h.hexdigest()[:16]

    def describe(self) -> str:
        return f"{self.name} ({self.maturity.value}, from {self.source})"


@dataclass
class SkillState:
    """A skill plus everything the gate decided about it."""

    skill: Skill
    status: Status
    reason: str = ""
    selftest_output: str = ""
    approved_hash: str | None = None
    findings: list[str] = field(default_factory=list)

    @property
    def loadable(self) -> bool:
        return self.status.loadable

    def to_dict(self) -> dict[str, Any]:
        """Everything a surface needs to render this skill.

        The single serialiser for all three interfaces. The CLI's ``--json``,
        the web API and the VSCode chat command all read this, because two
        hand-rolled serialisers is how the surfaces drifted apart the last
        time — a field added for one silently missing from the others.
        """
        # Imported lazily: ``scaffold`` pulls in introspection machinery that
        # a plain listing has no reason to load, and ``base`` must stay
        # importable on its own.
        try:
            from core.skills.scaffold import selftest_is_generated

            generated = selftest_is_generated(self.skill.path)
        except Exception:  # noqa: BLE001 - a listing must not fail over this
            generated = False
        try:
            from core.skills.layers import domains_of

            domains = domains_of(self.skill)
        except Exception:  # noqa: BLE001
            domains = []

        return {
            "name": self.skill.name,
            "status": self.status.value,
            "loadable": self.loadable,
            "maturity": self.skill.maturity.value,
            "kind": self.skill.kind.value,
            "source": self.skill.source,
            "path": str(self.skill.path),
            "content_hash": self.skill.content_hash(),
            "approved_hash": self.approved_hash,
            "reason": self.reason,
            "findings": self.findings,
            # A generated self-test proves the tooling runs, not that it
            # behaves. Every surface has to be able to say so, or a weaker
            # green tick looks identical to a stronger one.
            "selftest_generated": generated,
            "domains": domains,
            "scripts": self.skill.bundled_scripts(),
            "references": self.skill.reference_files(),
        }
