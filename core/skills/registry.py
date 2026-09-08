"""Finding skills, running their self-tests, and deciding what may load.

**FI ships no skills.** Not one, including any FI's own authors wrote.
A skill that arrives with the install is a curated library again, and
curation does not compound — the whole point is that capability is
*acquired*, from work done with this user, on this machine. So the
discovery root is user state, not repository content, and a fresh clone
starts with nothing.

Discovery has two sources, in this order:

1. ``<skills root>/<name>/`` — hand-authored and distilled skills. The
   root is ``FI_SKILLS_DIR``, else ``~/.frontier-insight/skills``. It is
   deliberately outside the repo: skills accumulate per machine and per
   person, and a skills directory inside the checkout would turn them
   into things you commit and ship.
2. The ``fi.skills`` entry-point group — skills installed from a package,
   so a library can ship its own skill without anyone editing FI. This is
   the reason the surface is pluggable at all: acquisition that requires
   a commit to FI does not compound.

A filesystem skill shadows an entry-point skill of the same name, so a
local edit always wins over an installed one.

The gate runs the self-test in a subprocess. That is not a sandbox and is
not claimed to be one — the same trust level as the venv executor FI
already uses for generated experiments. What it buys is that a self-test
which hangs, crashes the interpreter, or calls ``sys.exit`` cannot take
the quest down with it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from core.skills import approval
from core.skills.base import SELFTEST_PY, Skill, SkillState, Status

_log = logging.getLogger("fi.skills")

ENTRY_POINT_GROUP = "fi.skills"

#: A self-test is a fast sanity check on known-good values, not a
#: benchmark. Anything slower is a defect in the test.
SELFTEST_TIMEOUT_S = 120


_ENV_SKILLS_DIR = "FI_SKILLS_DIR"


def skills_root() -> Path:
    """Where this machine's skills live.

    Outside the repository on purpose. Skills are what FI has learned
    working with one person on one machine; putting them in the checkout
    would make them something you commit, review and ship — which is the
    curated-library model this design exists to replace.
    """
    override = os.environ.get(_ENV_SKILLS_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~")
    return Path(base) / ".frontier-insight" / "skills"


def _from_filesystem(skills_dir: Path) -> list[Skill]:
    out: list[Skill] = []
    try:
        entries = sorted(p for p in skills_dir.iterdir() if p.is_dir())
    except OSError:
        return out
    for d in entries:
        s = Skill(name=d.name, path=d, source="filesystem")
        if s.valid:
            out.append(s)
        else:
            _log.debug("skills: %s has no SKILL.md; not a skill", d)
    return out


def _from_entry_points() -> list[Skill]:
    """Skills advertised by installed packages.

    Best-effort by design: a broken third-party package must not stop FI
    from finding the skills that do load.
    """
    out: list[Skill] = []
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - importlib shape varies across versions
        return out

    for ep in eps:
        try:
            target = ep.load()
        except Exception as e:  # noqa: BLE001
            _log.warning("skills: entry point %r failed to load: %s", ep.name, e)
            continue
        path = Path(getattr(target, "__file__", "") or "").parent \
            if not isinstance(target, (str, Path)) else Path(target)
        s = Skill(name=ep.name, path=path, source=ENTRY_POINT_GROUP)
        if s.valid:
            out.append(s)
        else:
            _log.warning(
                "skills: entry point %r resolved to %s with no SKILL.md", ep.name, path,
            )
    return out


def discover(skills_dir: Path | None = None) -> list[Skill]:
    """Every skill FI can see. Filesystem shadows entry points by name."""
    fs = _from_filesystem(skills_dir or skills_root())
    seen = {s.name for s in fs}
    return fs + [s for s in _from_entry_points() if s.name not in seen]


def run_selftest(skill: Skill, *, timeout_s: int = SELFTEST_TIMEOUT_S) -> tuple[bool, str]:
    """Run the skill's own check. Returns (passed, combined output).

    A missing self-test is not a pass — see ``evaluate``. This function
    only reports what happened when one exists.
    """
    script = skill.file(SELFTEST_PY)
    if not script.is_file():
        return False, "no selftest.py"

    env = dict(os.environ)
    # The skill directory on the path so selftest.py can import skill.py
    # beside it without the skill needing to be an installed package.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(skill.path), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(skill.path),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"selftest timed out after {timeout_s}s"
    except OSError as e:
        return False, f"selftest could not run: {e}"

    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()[-4000:]


def _scan_findings(skill: Skill) -> list[str]:
    """Static findings for this skill, as rendered lines.

    Best-effort: the scanner is a review aid, and a bug in it must never
    stop a working skill from being evaluated. A skill with no findings and
    a skill whose scan crashed are different, so the second says so.
    """
    try:
        from core.skills import scan as _scan

        return [f.render() for f in _scan.scan(skill)]
    except Exception as e:  # noqa: BLE001
        _log.warning("skills: scan of %s failed: %s", skill.name, e)
        return [f"scan did not run ({e}) — review this skill by hand"]


def evaluate(
    skill: Skill,
    *,
    ledger: Path | None = None,
    run_test: bool = True,
) -> SkillState:
    """Decide what state a skill is in.

    Both gates must pass before a skill is loadable, and they are checked
    in this order because a failing self-test should be reported as a
    broken skill rather than an unapproved one:

    1. The self-test it carries must exist and pass.
    2. A person must have approved *this exact content*.

    ``run_test=False`` skips execution and reports on approval alone —
    used by listing commands that should not execute anything.

    The static scan runs on every path, including the ones that return
    early. This is the chokepoint every skill passes through — import,
    hand-authored, and entry-point alike — so scanning only at import would
    leave the other two routes unreviewed. Findings never change the status:
    they are attached for whoever is about to approve, because the rules are
    heuristics and QUARANTINED means *FI observed a failure*, not *FI
    guessed at intent*.
    """
    content_hash = skill.content_hash()
    approved = approval.approved_hash(skill.name, ledger)
    findings = _scan_findings(skill)

    if not skill.has_selftest:
        return SkillState(
            skill=skill,
            status=Status.UNTESTED,
            reason=(
                "no selftest.py — a skill that cannot demonstrate it still "
                "works is never promoted"
            ),
            approved_hash=approved,
            findings=findings,
        )

    output = ""
    if run_test:
        passed, output = run_selftest(skill)
        if not passed:
            return SkillState(
                skill=skill,
                status=Status.QUARANTINED,
                reason="selftest failed",
                selftest_output=output,
                approved_hash=approved,
                findings=findings,
            )

    if approved is None:
        reason = "awaiting approval — no person has approved this skill"
    elif approved != content_hash:
        reason = (
            f"content changed since approval (approved {approved}, "
            f"now {content_hash}) — re-approval required"
        )
    else:
        return SkillState(
            skill=skill,
            status=Status.TRUSTED,
            reason="selftest passes and this content is approved",
            selftest_output=output,
            approved_hash=approved,
            findings=findings,
        )

    return SkillState(
        skill=skill,
        status=Status.PROPOSED,
        reason=reason,
        selftest_output=output,
        approved_hash=approved,
        findings=findings,
    )


def loadable_skills(
    names: list[str],
    *,
    skills_dir: Path | None = None,
    ledger: Path | None = None,
) -> tuple[list[SkillState], list[SkillState]]:
    """Resolve requested skill names into (loadable, rejected).

    Requesting a skill that is not trusted is never an error — the quest
    proceeds with guided generation instead. Silently *using* an
    unapproved skill would be the error, so the rejects are returned
    rather than dropped, for the caller to log.
    """
    found = {s.name: s for s in discover(skills_dir)}
    ok: list[SkillState] = []
    rejected: list[SkillState] = []

    for name in names:
        skill = found.get(name)
        if skill is None:
            rejected.append(
                SkillState(
                    skill=Skill(name=name, path=Path(name), source="missing"),
                    status=Status.UNTESTED,
                    reason="no skill by that name was found",
                )
            )
            continue
        state = evaluate(skill, ledger=ledger)
        (ok if state.loadable else rejected).append(state)
    return ok, rejected
